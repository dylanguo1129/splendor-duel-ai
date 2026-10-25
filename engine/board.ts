// Board geometry: cell = (row-1)*5 + (col-1), rows/cols 1..5 as in BGA.

export function cellOf(row: number, col: number): number {
  return (row - 1) * 5 + (col - 1);
}
export function rowOf(cell: number): number {
  return Math.floor(cell / 5) + 1;
}
export function colOf(cell: number): number {
  return (cell % 5) + 1;
}

// Spiral fill order (BGA locationArg 1..25), center first.
const SPIRAL_RC: [number, number][] = [
  [3, 3], [4, 3], [4, 2], [3, 2], [2, 2], [2, 3], [2, 4], [3, 4], [4, 4], [5, 4],
  [5, 3], [5, 2], [5, 1], [4, 1], [3, 1], [2, 1], [1, 1], [1, 2], [1, 3], [1, 4],
  [1, 5], [2, 5], [3, 5], [4, 5], [5, 5],
];
export const SPIRAL: number[] = SPIRAL_RC.map(([r, c]) => cellOf(r, c));

// All straight runs of 1..3 adjacent cells (row, column, both diagonals).
// Order: 25 singles, then pairs, then triples.
function buildLines(): number[][] {
  const singles: number[][] = [];
  const pairs: number[][] = [];
  const triples: number[][] = [];
  for (let c = 0; c < 25; c++) singles.push([c]);
  const dirs: [number, number][] = [[0, 1], [1, 0], [1, 1], [1, -1]];
  for (const len of [2, 3]) {
    for (let r = 1; r <= 5; r++) {
      for (let c = 1; c <= 5; c++) {
        for (const [dr, dc] of dirs) {
          const cells: number[] = [];
          for (let k = 0; k < len; k++) {
            const rr = r + dr * k;
            const cc = c + dc * k;
            if (rr < 1 || rr > 5 || cc < 1 || cc > 5) break;
            cells.push(cellOf(rr, cc));
          }
          if (cells.length === len) (len === 2 ? pairs : triples).push(cells);
        }
      }
    }
  }
  return [...singles, ...pairs, ...triples];
}

export const LINES: number[][] = buildLines();

// Lookup: sorted cell key -> line index.
const lineIndex = new Map<string, number>();
LINES.forEach((cells, i) => lineIndex.set(cells.slice().sort((a, b) => a - b).join(","), i));

export function lineIndexOf(cells: number[]): number {
  const i = lineIndex.get(cells.slice().sort((a, b) => a - b).join(","));
  return i === undefined ? -1 : i;
}
