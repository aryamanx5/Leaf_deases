export const PALETTE_EVENT = 'patra:palette';
/** Opens the command palette from anywhere (the nav button uses it). */
export const openPalette = () => window.dispatchEvent(new Event(PALETTE_EVENT));
