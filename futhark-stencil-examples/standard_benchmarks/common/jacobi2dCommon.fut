-- A jacobi stencil is a weighted mean of the neighbours in a specific pattern.
import "./edgeHandling"

-- code is based on
-- https://gitlab.com/larisa.stoltzfus/liftstencil-cgo2018-artifact/-/blob/master/benchmarks/figure8/workflow2/j2d9pt/small/j2d9pt.c
-- However we chose to use uniform weights (aka. taking the mean)
-- , as opposed to the implementation in the reference code.

let mean_5points
    (p: (f32,f32,f32,f32,f32))
    : f32 = (p.0+p.1+p.2+p.3+p.4) / 5
let mean_9points
    (p: (f32,f32,f32,f32,f32,f32,f32,f32,f32))
    : f32 = (p.0+p.1+p.2+p.3+p.4+p.5+p.6+p.7+p.8) / 9

let single_iteration_maps_5points [Ny][Nx] (arr:[Ny][Nx]f32) =
  let bound = edgeHandling.extendEdge2D arr (Ny-1) (Nx-1)
  in tabulate_2d Ny Nx (\y x ->
        let rbound = bound y x
        let n = rbound (-1) ( 0)
        let w = rbound ( 0) (-1)
        let c = rbound ( 0) ( 0)
        let e = rbound ( 0) ( 1)
        let s = rbound ( 1) ( 0)
        in mean_5points (n,w,c,e,s)
        )

let single_iteration_stencil_5points arr =
  let ixs = [(-1,0),(0,-1),(0,0),(0,1),(1,0)] in
  let f _ v = mean_5points (v[0],v[1],v[2],v[3],v[4]) in
  let empty = map (map (const ())) arr in
  stencil_2d ixs f empty arr

let single_iteration_maps_9points [Ny][Nx] (arr:[Ny][Nx]f32) =
  let bound = edgeHandling.extendEdge2D arr (Ny-1) (Nx-1)
  in tabulate_2d Ny Nx (\y x ->
        let rbound = bound y x
        let n1 = rbound (-2) ( 0)
        let n2 = rbound (-1) ( 0)
        let w1 = rbound ( 0) (-2)
        let w2 = rbound ( 0) (-1)
        let c  = rbound ( 0) ( 0)
        let e1 = rbound ( 0) ( 1)
        let e2 = rbound ( 0) ( 2)
        let s1 = rbound ( 1) ( 0)
        let s2 = rbound ( 2) ( 0)
        in mean_9points (n1,n2,w1,w2,c,e1,e2,s1,s2)
        )

let single_iteration_stencil_9points arr =
  let ixs = [(-2,0),(-1,0),(0,-2),(0,-1)
            ,(0,0)
            ,(0,1),(0,2),(1,0),(2,0)] in
  let f _ v = mean_9points (v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8]) in
  let empty = map (map (const ())) arr in
  stencil_2d ixs f empty arr

let num_iterations: i32 = 5
let compute_iters [Ny][Nx] f (arr:[Ny][Nx]f32)
  : [Ny][Nx]f32 =
  iterate num_iterations f arr

module jacobi2dCommon = {
  let bench_5p_maps    = compute_iters single_iteration_maps_5points
  let bench_5p_stencil = compute_iters single_iteration_stencil_5points
  let bench_9p_maps    = compute_iters single_iteration_maps_9points
  let bench_9p_stencil = compute_iters single_iteration_stencil_9points
}
