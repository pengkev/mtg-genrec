"""Native dialog behavior: focus trapping, Escape, backdrop close and copy action."""
DIALOG_TEMPLATE = '<dialog aria-labelledby="card-dialog-title"><div class="card-dialog-content">${value}</div></dialog>'
DIALOG_JS = r"""
let previousFocus = null;
let parentViewport = null;
const embedded = window.parent !== window && window.parentIFrame?.getPageInfo;
function positionDialog(dialog) {
    if (!parentViewport) return;
    const info = parentViewport;
    const top = Math.max(0, info.scrollTop - info.offsetTop);
    const bottom = Math.min(info.iframeHeight, info.scrollTop - info.offsetTop + info.windowHeight);
    const left = Math.max(0, info.scrollLeft - info.offsetLeft);
    const right = Math.min(info.iframeWidth, info.scrollLeft - info.offsetLeft + info.windowWidth);
    dialog.style.setProperty('--dialog-center-y', `${(top + bottom) / 2}px`);
    dialog.style.setProperty('--dialog-center-x', `${(left + right) / 2}px`);
    dialog.style.setProperty('--dialog-viewport-height', `${Math.max(0, bottom - top)}px`);
}
// Hugging Face expands the iframe to the document height. Follow the visible
// intersection with the parent viewport, including parent scrolling/resizing.
if (embedded) {
    window.parentIFrame.getPageInfo(info => {
        parentViewport = info;
        const dialog = element.querySelector('dialog');
        if (dialog) positionDialog(dialog);
        if (props.value && !dialog?.open) syncDialog();
    });
}
const oracleCache = new Map();
function symbolText(text) {
    const fragment = document.createDocumentFragment();
    for (const part of String(text || '').split(/(\{[A-Z0-9/]+\}|\n)/g)) {
        if (part === '\n') fragment.append(document.createElement('br'));
        else if (/^\{[A-Z0-9/]+\}$/.test(part)) {
            const img = document.createElement('img');
            img.className = 'mana-symbol';
            img.src = 'https://svgs.scryfall.io/card-symbols/' + part.slice(1, -1).replaceAll('/', '') + '.svg';
            img.alt = part;
            fragment.append(img);
        } else fragment.append(document.createTextNode(part));
    }
    return fragment;
}
async function refreshOracle(dialog) {
    const header = dialog.querySelector('[data-card-name]');
    if (!header) return;
    const name = header.dataset.cardName;
    try {
        if (!oracleCache.has(name)) {
            oracleCache.set(name, fetch('https://api.scryfall.com/cards/named?exact=' + encodeURIComponent(name))
                .then(response => { if (!response.ok) throw new Error('Oracle lookup failed'); return response.json(); }));
        }
        const card = await oracleCache.get(name);
        if (dialog.querySelector('[data-card-name]') !== header) return;
        const faces = card.card_faces || [card];
        dialog.querySelectorAll('.oracle-text').forEach(node => {
            const face = faces[Number(node.dataset.face)];
            if (face) node.replaceChildren(symbolText(face.oracle_text ?? card.oracle_text ?? ''));
        });
    } catch { oracleCache.delete(name); } // Retain the catalog text when offline.
}
function syncDialog() {
    const dialog = element.querySelector('dialog');
    if (!dialog) return;
    positionDialog(dialog);
    if (props.value) {
        if (embedded && !parentViewport) return;
        refreshOracle(dialog);
        if (!dialog.open) {
            previousFocus = document.activeElement;
            dialog.showModal();
            dialog.querySelector('[data-close]')?.focus({ preventScroll: true });
        }
    } else if (dialog.open) dialog.close();
}
function closeDialog() {
    element.querySelector('dialog')?.close();
    previousFocus?.focus({ preventScroll: true });
}
element.addEventListener('click', (event) => {
    const dialog = element.querySelector('dialog');
    if (event.target.closest('[data-close]')) closeDialog();
    if (event.target === dialog) {
        const bounds = dialog.getBoundingClientRect();
        if (event.clientX < bounds.left || event.clientX > bounds.right ||
            event.clientY < bounds.top || event.clientY > bounds.bottom) closeDialog();
    }
    const button = event.target.closest('[data-add]');
    if (button && !button.disabled) {
        const message = dialog.querySelector('.modal-message');
        if (message) message.textContent = 'Adding…';
        trigger('add_copies', {quantity: 1});
    }
});
// Delegation survives HTML updates after selecting or adding a card.
element.addEventListener('cancel', () => previousFocus?.focus({ preventScroll: true }), true);
watch('value', syncDialog);
syncDialog();
"""
# Gradio's HTML component scopes its CSS, so include this in both stylesheets.
SCROLLBAR_CSS = """
* {
    scrollbar-width: none !important;
    -ms-overflow-style: none !important;
}
*::-webkit-scrollbar {
    display: none !important;
    width: 0 !important;
    height: 0 !important;
}
"""
DIALOG_CSS = SCROLLBAR_CSS + """
dialog { position: fixed !important; inset: auto !important;
    top: var(--dialog-center-y, 50%) !important; left: var(--dialog-center-x, 50%) !important; transform: translate(-50%, -50%);
    margin: 0 !important;
    box-sizing: border-box; width: min(900px, calc(100vw - 32px));
    height: fit-content; max-width: calc(100vw - 32px); max-height: calc(var(--dialog-viewport-height, 100dvh) - 40px);
    border: 0; border-radius: 20px; box-shadow: 0 24px 80px #0006;
    background: var(--background-fill-primary); color: var(--body-text-color);
    padding: 28px; overflow-y: auto; overscroll-behavior: contain; }
dialog::backdrop { background: rgba(0, 0, 0, .72); backdrop-filter: blur(5px); }
header { display: flex; align-items: start; justify-content: space-between; gap: 20px; margin-bottom: 24px; }
h2 { font-size: 1.6rem; line-height: 1.2; margin: 0; }
h3 { font-size: 1.1rem; font-weight: 600; }
button { cursor: pointer; border: 0; padding: 12px 20px; border-radius: 10px; font-weight: 600; }
[data-close] { background: transparent; color: inherit; padding: 4px 10px; font-size: 1.2rem; }
button:focus-visible { outline: 3px solid var(--color-accent); outline-offset: 3px; }
button:disabled { cursor: default; opacity: .5; }
[data-add] { background: var(--button-primary-background-fill); color: var(--button-primary-text-color); }
.card-face { display: grid; grid-template-columns: minmax(180px, 310px) minmax(0, 1fr); gap: 32px; margin-bottom: 24px; align-items: start; }
.card-face.no-art { grid-template-columns: 1fr; }
.card-face > img { width: 100%; height: auto; border-radius: 14px; }
.mana-symbol { display: inline-block; width: 1.15em; height: 1.15em; vertical-align: -.2em; margin: 0 .08em; }
.mana-cost { font-size: 1.2rem; }
p { margin: 0 0 16px; line-height: 1.7; }
.oracle-text { font-size: 1.05rem; }
footer { display: flex; flex-wrap: wrap; gap: 16px; align-items: center; padding-top: 8px; }
.modal-message { margin: 0; font-size: .9rem; }
@media (max-width: 600px) {
    dialog { padding: 20px; }
    .card-face { grid-template-columns: 1fr; gap: 20px; }
    .card-face > img { width: min(280px, 75vw); margin: auto; }
}
"""
APP_CSS = SCROLLBAR_CSS + """
#recommendation-gallery { border: 0; background: transparent; box-shadow: none; }
#recommendation-gallery .grid-wrap { padding: 4px; max-height: none; overflow: visible; }
#recommendation-gallery .grid-container {
    grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)) !important;
    grid-template-rows: none !important; grid-auto-rows: auto !important; gap: 24px 20px;
}
#recommendation-gallery .thumbnail-item.thumbnail-lg {
    aspect-ratio: auto; border: 0; background: transparent; box-shadow: none;
    overflow: visible; height: auto; padding: 0; display: flex; flex-direction: column;
}
#recommendation-gallery .thumbnail-lg > img {
    width: 100%; height: auto; aspect-ratio: 5 / 7; object-fit: contain;
    border-radius: 12px; transition: transform .15s ease;
}
#recommendation-gallery .thumbnail-lg:hover > img { transform: translateY(-4px); }
#recommendation-gallery .thumbnail-lg:focus-visible { outline: 3px solid var(--color-accent); outline-offset: 5px; }
#recommendation-gallery .caption-label {
    position: static; max-width: 100%; width: 100%; padding: 10px 0 0;
    border: 0; background: transparent; white-space: normal; line-height: 1.4;
    font-size: .9rem; text-align: left; opacity: 1 !important;
}
@media (max-width: 640px) {
    #recommendation-gallery .grid-container { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; gap: 20px 12px; }
}
"""

# Update only the live confirmation node after the server confirms the result.
ADD_CONFIRMATION_JS = r"""(message) => {
    const dialog = document.querySelector('#card-modal dialog[open]');
    const confirmation = dialog?.querySelector('.modal-message');
    if (confirmation) {
        confirmation.textContent = message;
        confirmation.animate(
            [{ opacity: 0.35 }, { opacity: 1 }],
            { duration: 250, easing: 'ease-out' }
        );
    }
}"""
