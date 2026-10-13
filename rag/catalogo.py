"""Acceso al catalogo de servicios como FICHAS completas (no fragmentos).

La busqueda hibrida devuelve fragmentos sueltos; pero cuando el especialista
quiere ver un servicio entero, o cuando el codigo necesita validar un
servicio_id y completar su tarifa, hace falta la ficha completa. Se arma
agrupando los fragmentos de Chroma por servicio_id, asi Chroma sigue siendo
la unica fuente de verdad.

Las fichas se cargan una vez por proceso. Si se edita el corpus y se
re-indexa, hay que reiniciar el worker (lo hace solo `docker compose restart`).
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from rag.almacen import AlmacenChroma


class FichaServicio(BaseModel):
    servicio_id: str
    servicio: str
    prioridad: str
    estado: str
    tipo: str
    tarifa: str
    texto: str

    @property
    def ofrecible(self) -> bool:
        return self.estado == "activo"


class Catalogo:
    def __init__(self, almacen: AlmacenChroma) -> None:
        self._almacen = almacen
        self._fichas: dict[str, FichaServicio] | None = None
        self._candado = asyncio.Lock()

    async def _cargar(self) -> dict[str, FichaServicio]:
        if self._fichas is not None:
            return self._fichas
        async with self._candado:
            if self._fichas is None:
                fragmentos = sorted(await self._almacen.todos("catalogo"), key=lambda f: f.id)
                fichas: dict[str, FichaServicio] = {}
                for frag in fragmentos:
                    m = frag.metadatos
                    sid = m["servicio_id"]
                    if sid not in fichas:
                        fichas[sid] = FichaServicio(
                            servicio_id=sid, servicio=m["servicio"], prioridad=m["prioridad"],
                            estado=m["estado"], tipo=m.get("tipo", "servicio"), tarifa=m["tarifa"], texto="",
                        )
                    # Se saca el encabezado de contexto: en la ficha completa sobra.
                    cuerpo = frag.texto.split("\n\n", 1)[-1]
                    fichas[sid].texto += f"## {m['seccion']}\n{cuerpo}\n\n"
                self._fichas = fichas
        return self._fichas

    async def ficha(self, servicio_id: str) -> FichaServicio | None:
        return (await self._cargar()).get(servicio_id.strip())

    async def ids_ofrecibles(self) -> list[str]:
        return sorted(sid for sid, f in (await self._cargar()).items() if f.ofrecible)

    async def tarifas_validas(self) -> set[str]:
        """Todas las tarifas con precio del catalogo (para el chequeo de precios)."""
        return {f.tarifa for f in (await self._cargar()).values() if any(c.isdigit() for c in f.tarifa)}
