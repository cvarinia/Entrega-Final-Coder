"""Almacen vectorial: ChromaDB corriendo como servidor (Docker).

Por que servidor y no `PersistentClient` (lo que se uso en los ejercicios):
    - PersistentClient es SINCRONICO: cada consulta bloquearia el event loop
      del worker, justo el error que advierte la consigna.
    - Con el servidor se usa `AsyncHttpClient`, el cliente asincrono oficial.
    - La API, el worker y el script de indexado comparten la misma base.

Una coleccion por cada parte del corpus:
    <prefijo>_catalogo     y     <prefijo>_metodologia
El prefijo viene del .env (CHROMA_PREFIJO_COLECCIONES): ningun nombre de
indice queda escrito en el codigo.

Indexado idempotente: la coleccion guarda en sus metadatos la "huella" del
corpus (hash del contenido + modelo de embeddings). Si al arrancar la huella
coincide, no se re-indexa ni se gasta una sola llamada a la API.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from comun.config import Configuracion
from rag.corpus import Coleccion, Fragmento, huella
from rag.embeddings import Vectorizador


class FragmentoRecuperado(BaseModel):
    """Un fragmento devuelto por una busqueda."""

    id: str
    texto: str
    metadatos: dict[str, Any]
    distancia: float | None = None  # solo en la busqueda vectorial


class AlmacenChroma:
    """Toda la interaccion con Chroma pasa por aca."""

    def __init__(self, cliente: Any, prefijo: str) -> None:
        self._cliente = cliente
        self._prefijo = prefijo

    @classmethod
    async def conectar(cls, config: Configuracion, intentos: int = 1) -> "AlmacenChroma":
        """Abre el cliente asincrono y verifica que el servidor responda.

        Con Docker Compose, el worker puede arrancar antes de que Chroma este
        listo: por eso el worker pide varios intentos (uno cada 2 segundos).
        """
        import asyncio

        import chromadb

        for intento in range(1, intentos + 1):
            try:
                cliente = await chromadb.AsyncHttpClient(host=config.chroma_host, port=config.chroma_port)
                await cliente.heartbeat()  # si Chroma no esta levantado, falla aca y claro
                return cls(cliente, config.chroma_prefijo)
            except Exception:
                if intento == intentos:
                    raise
                print(f"  Chroma todavia no responde ({config.chroma_host}:{config.chroma_port}), "
                      f"reintento {intento}/{intentos - 1}...", flush=True)
                await asyncio.sleep(2)
        raise RuntimeError("inalcanzable")

    def nombre(self, coleccion: Coleccion) -> str:
        return f"{self._prefijo}_{coleccion}"

    async def _coleccion(self, coleccion: Coleccion):
        # embedding_function=None: los vectores los calculamos nosotros (Gemini).
        # Si no, Chroma usaria su modelo por defecto (solo ingles) sin avisar.
        return await self._cliente.get_or_create_collection(
            name=self.nombre(coleccion),
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
        )

    # --- Indexado -------------------------------------------------------------

    async def indexar(
        self,
        coleccion: Coleccion,
        fragmentos: list[Fragmento],
        vectorizador: Vectorizador,
        modelo_embeddings: str,
        forzar: bool = False,
    ) -> bool:
        """Carga los fragmentos en la coleccion. Devuelve True si re-indexo.

        Si la huella guardada coincide con la del corpus actual, no hace nada.
        """
        huella_actual = huella(fragmentos, modelo_embeddings)
        existente = await self._coleccion(coleccion)
        guardada = (existente.metadata or {}).get("huella")
        if guardada == huella_actual and not forzar:
            return False

        # Algo cambio: se reconstruye la coleccion desde cero (el corpus es
        # chico; borrar y recrear es mas simple y seguro que sincronizar).
        await self._cliente.delete_collection(self.nombre(coleccion))
        nueva = await self._cliente.create_collection(
            name=self.nombre(coleccion),
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"huella": huella_actual, "modelo_embeddings": modelo_embeddings},
        )
        vectores = await vectorizador.vectorizar_documentos([f.texto for f in fragmentos])
        await nueva.add(
            ids=[f.id for f in fragmentos],
            documents=[f.texto for f in fragmentos],
            embeddings=vectores,
            metadatas=[f.metadatos for f in fragmentos],
        )
        return True

    # --- Consultas ------------------------------------------------------------

    async def buscar_por_vector(
        self,
        coleccion: Coleccion,
        vector: list[float],
        cantidad: int,
        filtro: dict[str, Any] | None = None,
    ) -> list[FragmentoRecuperado]:
        """Los `cantidad` fragmentos mas cercanos al vector, respetando el filtro."""
        col = await self._coleccion(coleccion)
        resultado = await col.query(
            query_embeddings=[vector],
            n_results=cantidad,
            where=filtro or None,
            include=["documents", "metadatas", "distances"],
        )
        return [
            FragmentoRecuperado(id=i, texto=t, metadatos=m or {}, distancia=d)
            for i, t, m, d in zip(
                resultado["ids"][0],
                resultado["documents"][0],
                resultado["metadatas"][0],
                resultado["distances"][0],
            )
        ]

    async def todos(
        self, coleccion: Coleccion, filtro: dict[str, Any] | None = None
    ) -> list[FragmentoRecuperado]:
        """Todos los fragmentos que cumplen el filtro (los usa la busqueda por palabras)."""
        col = await self._coleccion(coleccion)
        resultado = await col.get(where=filtro or None, include=["documents", "metadatas"])
        return [
            FragmentoRecuperado(id=i, texto=t, metadatos=m or {})
            for i, t, m in zip(resultado["ids"], resultado["documents"], resultado["metadatas"])
        ]

    async def cantidad(self, coleccion: Coleccion) -> int:
        return await (await self._coleccion(coleccion)).count()
