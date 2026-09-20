// NOT vendored — a minimal local type shim, not the real crypto_sor `feedhandlers/index.ts`.
//
// The real vendored `CompositeOrderBook.ts` (`../lib/CompositeOrderBook.ts`) imports
// `OrderBookAction`/`OrderBookEvent` from this path only for `processExternalOrderBookEvent()`,
// a method this comparison's harness never calls (it drives the real, unmodified class directly
// through the real, public `updateLevel()`/`newOrder()` methods instead). These two declarations
// exist only so the real file's own import statement resolves and the real code it actually
// exercises compiles and runs unmodified — the two names below are shaped from reading the real
// file's own real usage (`action === OrderBookAction.Partial`, `event.bids`/`event.asks` as
// [price, size] tuple arrays), not invented.

export enum OrderBookAction {
    Partial,
    Update,
}

export interface OrderBookEvent {
    action: OrderBookAction
    symbol: string
    bids: [number, number][]
    asks: [number, number][]
}
