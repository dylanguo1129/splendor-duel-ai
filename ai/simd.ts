// Hand-assembled WebAssembly SIMD kernels for the network's dense layers (ai/net.ts uses them when
// available; plain JS otherwise). Roughly 2-3x faster than the JS loops, i.e. more simulations per
// second in the browser at the same thinking time.
//
//   layer(in, nIn, w, b, out, nOut, relu): out = b + sum_j in[j] * w[j][*] (zero inputs skipped,
//                                          w input-major, nOut % 4 == 0), then ReLU if relu != 0
//   dot(a, b, n) -> f32:                   sum_i a[i] * b[i] (n % 4 == 0)
// All pointers are byte offsets into the imported memory "env.mem" (16-byte aligned).

const uleb = (n: number): number[] => {
  const out: number[] = [];
  do {
    let b = n & 0x7f;
    n >>>= 7;
    if (n) b |= 0x80;
    out.push(b);
  } while (n);
  return out;
};
const sleb = (n: number): number[] => {
  const out: number[] = [];
  for (;;) {
    const b = n & 0x7f;
    n >>= 7;
    if ((n === 0 && !(b & 0x40)) || (n === -1 && b & 0x40)) return [...out, b];
    out.push(b | 0x80);
  }
};
const str = (s: string) => [...uleb(s.length), ...[...s].map((c) => c.charCodeAt(0))];
const section = (id: number, body: number[]) => [id, ...uleb(body.length), ...body];
const vec = (items: number[][]) => [...uleb(items.length), ...items.flat()];

// opcodes
const I32 = 0x7f, F32 = 0x7d, V128 = 0x7b;
const block = [0x02, 0x40], loop = [0x03, 0x40], if_ = [0x04, 0x40], end = [0x0b];
const br = (d: number) => [0x0c, d], brIf = (d: number) => [0x0d, d];
const get = (l: number) => [0x20, l], set = (l: number) => [0x21, l];
const i32c = (v: number) => [0x41, ...sleb(v)];
const i32add = [0x6a], i32shl = [0x74], i32geu = [0x4f], f32ne = [0x5c], f32add = [0x92];
const f32load = [0x2a, 0x02, 0x00];
const f32zero = [0x43, 0, 0, 0, 0];
const simd = (op: number, ...imm: number[]) => [0xfd, ...uleb(op), ...imm];
const vload = simd(0x00, 0x04, 0x00), vstore = simd(0x0b, 0x04, 0x00);
const vzero = simd(0x0c, ...new Array(16).fill(0));
const splat = simd(0x13), lane = (k: number) => simd(0x1f, k);
const vadd = simd(0xe4), vmul = simd(0xe6), vmax = simd(0xe9);

/** for (i = 0; i < bytes; i += 16) body  (i, bytes: locals; `body` may use depth-relative branches) */
const forBytes = (i: number, bytes: number, body: number[]) => [
  ...i32c(0), ...set(i),
  ...block, ...loop,
  ...get(i), ...get(bytes), ...i32geu, ...brIf(1),
  ...body,
  ...get(i), ...i32c(16), ...i32add, ...set(i),
  ...br(0),
  ...end, ...end,
];

function layerBody(): number[] {
  // params: in 0, nIn 1, w 2, b 3, out 4, nOut 5, relu 6; locals: i 7, j 8, row 9, bytes 10, x 11 (f32), xv 12 (v128)
  const [IN, NIN, W, B, OUT, NOUT, RELU, I, J, ROW, BYTES, X, XV] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
  const at = (base: number) => [...get(base), ...get(I), ...i32add];
  const code = [
    ...get(NOUT), ...i32c(2), ...i32shl, ...set(BYTES),
    // out = b
    ...forBytes(I, BYTES, [...at(OUT), ...at(B), ...vload, ...vstore]),
    // out += in[j] * w[j] for non-zero in[j]
    ...i32c(0), ...set(J),
    ...get(W), ...set(ROW),
    ...block, ...loop,
    ...get(J), ...get(NIN), ...i32geu, ...brIf(1),
    ...get(IN), ...get(J), ...i32c(2), ...i32shl, ...i32add, ...f32load, ...set(X),
    ...get(X), ...f32zero, ...f32ne,
    ...if_,
    ...get(X), ...splat, ...set(XV),
    ...forBytes(I, BYTES, [...at(OUT), ...at(OUT), ...vload, ...get(ROW), ...get(I), ...i32add, ...vload, ...get(XV), ...vmul, ...vadd, ...vstore]),
    ...end,
    ...get(ROW), ...get(BYTES), ...i32add, ...set(ROW),
    ...get(J), ...i32c(1), ...i32add, ...set(J),
    ...br(0),
    ...end, ...end,
    // ReLU
    ...get(RELU),
    ...if_,
    ...forBytes(I, BYTES, [...at(OUT), ...at(OUT), ...vload, ...vzero, ...vmax, ...vstore]),
    ...end,
    ...end,
  ];
  const locals = vec([[...uleb(4), I32], [...uleb(1), F32], [...uleb(1), V128]]);
  return [...locals, ...code];
}

function dotBody(): number[] {
  // params: a 0, b 1, n 2; locals: i 3, bytes 4, acc 5 (v128)
  const [A, B, N, I, BYTES, ACC] = [0, 1, 2, 3, 4, 5];
  const code = [
    ...get(N), ...i32c(2), ...i32shl, ...set(BYTES),
    ...forBytes(I, BYTES, [...get(ACC), ...get(A), ...get(I), ...i32add, ...vload, ...get(B), ...get(I), ...i32add, ...vload, ...vmul, ...vadd, ...set(ACC)]),
    ...get(ACC), ...lane(0),
    ...get(ACC), ...lane(1), ...f32add,
    ...get(ACC), ...lane(2), ...f32add,
    ...get(ACC), ...lane(3), ...f32add,
    ...end,
  ];
  const locals = vec([[...uleb(2), I32], [...uleb(1), V128]]);
  return [...locals, ...code];
}

function moduleBytes(): Uint8Array {
  const types = vec([
    [0x60, ...vec(new Array(7).fill([I32])), 0x00],
    [0x60, ...vec([[I32], [I32], [I32]]), 0x01, F32],
  ]);
  const imports = vec([[...str("env"), ...str("mem"), 0x02, 0x00, 0x01]]); // memory, min 1 page
  const funcs = vec([[0x00], [0x01]]);
  const exports = vec([
    [...str("layer"), 0x00, 0x00],
    [...str("dot"), 0x00, 0x01],
  ]);
  const bodies = [layerBody(), dotBody()].map((b) => [...uleb(b.length), ...b]);
  return new Uint8Array([
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    ...section(1, types),
    ...section(2, imports),
    ...section(3, funcs),
    ...section(7, exports),
    ...section(10, vec(bodies)),
  ]);
}

export interface SimdKernels {
  layer(inPtr: number, nIn: number, w: number, b: number, out: number, nOut: number, relu: number): void;
  dot(a: number, b: number, n: number): number;
}

let compiled: WebAssembly.Module | null | undefined;

/** Kernels bound to `memory`, or null when WebAssembly SIMD is unavailable. */
export function simdKernels(memory: WebAssembly.Memory): SimdKernels | null {
  try {
    if (compiled === undefined) compiled = typeof WebAssembly === "object" ? new WebAssembly.Module(moduleBytes()) : null;
    if (!compiled) return null;
    return new WebAssembly.Instance(compiled, { env: { mem: memory } }).exports as unknown as SimdKernels;
  } catch {
    compiled = null;
    return null;
  }
}
