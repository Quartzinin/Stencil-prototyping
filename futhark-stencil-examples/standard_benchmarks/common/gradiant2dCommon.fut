-- Computes the mean of the finite difference appoximation
-- of all forwards and backwards derivatives.
import "./edgeHandling"

-- This code is based on a reference implementation found in:
-- https://gitlab.com/larisa.stoltzfus/liftstencil-cgo2018-artifact/-/blob/master/benchmarks/figure8/workflow1/grad2d/small/grad2d.c

let avoid_div_zero: f32 = 0.0001
let gradiant_5point ((N,W,C,E,S): (f32,f32,f32,f32,f32)) : f32 =
    let ns = let tmp = C-N in tmp*tmp in
    let ws = let tmp = C-W in tmp*tmp in
    let es = let tmp = C-E in tmp*tmp in
    let ss = let tmp = C-S in tmp*tmp in
    C + (1.0 / f32.sqrt (avoid_div_zero + ns + ws + es + ss))

let single_iteration_maps_5points [Ny][Nx] (arr:[Ny][Nx]f32) =
  let bound = edgeHandling.extendEdge2D arr (Ny-1) (Nx-1)
  in tabulate_2d Ny Nx (\y x ->
        let rbound = bound y x
        let N = rbound (-1) ( 0)
        let W = rbound ( 0) (-1)
        let C = rbound ( 0) ( 0)
        let E = rbound ( 0) ( 1)
        let S = rbound ( 1) ( 0)
        in gradiant_5point (N,W,C,E,S)
        )

-- There is actually only 1 iteration.
let single_iteration_stencil_5points arr =
  let ixs = [(-1, 0),( 0,-1),( 0, 0),( 0, 1),( 1, 0)] in
  let f _ v = gradiant_5point (v[0] ,v[1] ,v[2] ,v[3] ,v[4]) in
  let empty = map (map (const ())) arr in
  stencil_2d ixs f empty arr

module gradiantCommon = {
    let bench_5p_maps    = single_iteration_maps_5points
    let bench_5p_stencil = single_iteration_stencil_5points
}
