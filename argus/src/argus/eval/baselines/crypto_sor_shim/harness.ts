// Not vendored — this comparison's own thin JSON-stdin/JSON-stdout wrapper around the real,
// unmodified `CompositeOrderBook` class. Reads one JSON object from stdin:
//   { symbol, side: "BUY"|"SELL", orderQty, exchanges: string[]|null,
//     levels: [{exchange, side: "BUY"|"SELL", price, size}, ...] }
// Calls the real `updateLevel()` for every level, then the real, unmodified `newOrder()`, and
// prints the real `Execution[]` it returns as JSON — nothing about the fill logic is touched.

import { CompositeOrderBook } from './src/lib/CompositeOrderBook'
import { Side } from './src/lib/common'

let input = ''
process.stdin.on('data', (chunk) => { input += chunk })
process.stdin.on('end', () => {
    const req = JSON.parse(input)
    const book = new CompositeOrderBook(req.symbol)
    for (const lvl of req.levels) {
        const side = lvl.side === 'BUY' ? Side.Buy : Side.Sell
        book.updateLevel(lvl.exchange, side, lvl.price, lvl.size)
    }
    const side = req.side === 'BUY' ? Side.Buy : Side.Sell
    const executions = book.newOrder(side, req.orderQty, req.exchanges ?? undefined)
    process.stdout.write(JSON.stringify(executions))
})
