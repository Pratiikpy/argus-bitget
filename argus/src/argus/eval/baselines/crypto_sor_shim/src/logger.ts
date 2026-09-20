// NOT vendored — a minimal local no-op shim, not the real crypto_sor `logger.ts` (which wraps
// the real `pino` package). The real vendored `CompositeOrderBook.ts` calls `logger.info(...)`
// only inside `vacuum()`, a method this comparison's harness never calls. A no-op default export
// with an `info()` method is all the real file's own import statement needs to resolve.

export default {
    info: (..._args: unknown[]): void => {},
}
