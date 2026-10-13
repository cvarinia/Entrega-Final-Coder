# Pruebas del sistema — `pruebas-20261013-103458`

| # | Agencia | Estado | Completa | Recorrido | Puntaje | Aprobaciones | Canal | Duración (s) |
|---|---|---|---|---|---|---|---|---|
| 1 | Vistatec | DONE | True | investigador → especialista → redactor → FINALIZAR | 8 | v1 aprobado | portal_proveedores | 66.66 |
| 2 | Welocalize | DONE | True | investigador → especialista → redactor → especialista → redactor → FINALIZAR | 10 | v1 rechazado; v2 aprobado | portal_proveedores | 96.84 |
| 3 | Alpha CRC | DONE | True | investigador → especialista → redactor → FINALIZAR | 9 | v1 aprobado | portal_proveedores | 69.3 |
| 4 | SeproTec | DONE | True | investigador → especialista → redactor → FINALIZAR | 11 | v1 aprobado | email | 69.5 |
| 5 | Lionbridge | DONE | True | investigador → especialista → redactor → FINALIZAR | 8 | v1 aprobado | portal_proveedores | 62.71 |

**Pedido inválido** (`agencia: "<nombre>"`): HTTP 422 ✅ rechazado por Pydantic

## Métricas (Phoenix)

- Trazas: 11
- Costo total: US$ 0.6486 — por ejecución: US$ 0.0590
- Tokens: 505126 entrada / 28704 salida
- Latencia p50: 16.1 s — p95: 63.4 s
- Filtro en Phoenix: `metadata['etiqueta'] == 'pruebas-20261013-103458'`

## Mensajes generados

### Vistatec (portal_proveedores, ingles)

**Asunto:** Storyline & Rise localization for e-learning  
```
Hi Vistatec team,

We noticed your e-learning localization service supports 150+ languages with cultural adaptation. We specialize in Storyline and Rise course localization—handling complex .story files, layout adjustments for RTL and Asian languages, and multilingual DTP integration.

With over 10 years in e-learning localization, we work as a single point of contact for your team, reducing coordination overhead. Our rate is 14 USD/h for Storyline, Rise, and multilingual DTP.

Would a short test project in Storyline or Rise help demonstrate our fit for your e-learning pipeline?
```

### Welocalize (portal_proveedores, ingles)

**Asunto:** Storyline localization and DTP support for Welocalize  
```
Hi Welocalize team,

We noticed you work across 300+ languages and specialize in e-learning localization with Articulate Storyline. We're a small studio with over 10 years in Storyline localization and multilingual DTP—we handle course formatting and text layout across any language pair.

Instead of coordinating multiple freelancers, you'd have a single point of contact for both services. Our rates are $14/h for Storyline localization and $14/h for DTP work.

Would a short test project in Storyline help demonstrate how we could fit into your e-learning pipeline?
```

### Alpha CRC (portal_proveedores, ingles)

**Asunto:** Multilingual DTP and Articulate Localization Support  
```
Alpha CRC manages e-learning localization across 15 countries in complex languages—Chinese, Arabic, Cyrillic. We specialize in multilingual DTP and Articulate localization (Storyline and Rise), handling layout challenges that often break after translation. With over 10 years in e-learning, we work as a single point of contact for your team, reducing coordination overhead. All services at 14 USD/h. Would a short test project in Storyline or Rise help demonstrate our fit for your pipeline?
```

### SeproTec (email, español)

**Asunto:** Localización Storyline y Rise para SeproTec  
```
Hola,

Vemos que SeproTec trabaja con Articulate Storyline en 220 idiomas y está en expansión. Headway es un estudio especializado en localización de cursos de e-learning (Storyline, Rise) y DTP multilingüe. Resolvemos la maquetación técnica en idiomas asiáticos, RTL y cirílico, sin depender de proveedores externos. Trabajamos a 14 USD/h con un solo interlocutor para tu equipo.

¿Te gustaría que hagamos una prueba corta en Storyline o Rise para ver cómo encajamos en tu pipeline?
```

### Lionbridge (portal_proveedores, ingles)

**Asunto:** Figma and eLearning localization for Lionbridge  
```
Hi,

We noticed Lionbridge is expanding its eLearning and multimedia localization services across 14+ verticals. We specialize in localizing interactive prototypes in Figma, Storyline courses, Rise modules, and multilingual DTP—services that complement your current pipeline.

With over 10 years in e-learning localization, we work as a single point of contact for your team, reducing coordination overhead. Our rates are $14/h for Storyline and Rise, with custom quotes for Figma projects.

Would a short test project help demonstrate our fit for your e-learning and design localization needs?
```
