# Chat template provenance

`chat_template.jinja` is an unmodified copy of
[MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/blob/main/files/chat_template.jinja),
retrieved on 2026-10-08 at the operator's request.

SHA-256: `7a5a0dda1331a7c40d930961cc1cb3b57c3b52625250c13372fe006ba2e9dfdb`.
Upstream attribution and applicable terms remain with the source project.

The file is loaded from `/deployment/files/chat_template.jinja` through the
read-only deployment mount. The previous image template remains available at
`/usr/local/share/glm53-chat-template.jinja` for rollback. Both nodes must use
matching template bytes and `CHAT_TEMPLATE`.

The matching `files/overlays/fixes/glm47_moe.py` parser overlay honours the
template's thinking flags. The image's previous parser forces reasoning on;
without the overlay, disabled-thinking answers can land in `reasoning` with
empty `content`. To restore the earlier low-effort behaviour, restore the
pre-change deployment backup, including Compose and the previous template path.
