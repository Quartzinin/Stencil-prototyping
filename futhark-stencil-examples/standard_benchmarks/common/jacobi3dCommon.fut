import "./edgeHandling"

-- code is based on
-- https://gitlab.com/larisa.stoltzfus/liftstencil-cgo2018-artifact/-/blob/master/benchmarks/figure7/workflow1/reference/hotspot3D/hotspotKernel.cl
--

let mean_7points
    (p: (f32,f32,f32,f32,f32,f32,f32))
    : f32 = (p.0+p.1+p.2+p.3+p.4+p.5+p.6) / 7
let mean_13points
    (p: (f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32))
    : f32 = (p.0+p.1+p.2+p.3+p.4+p.5+p.6+p.7+p.8+p.9+p.10+p.11+p.12) / 13

let single_iteration_maps_7points [Nz][Ny][Nx] (arr:[Nz][Ny][Nx]f32) =
  let bound = edgeHandling.extendEdge3D arr (Nz-1) (Ny-1) (Nx-1)
  in tabulate_3d Nz Ny Nx (\z y x ->
        let rbound = bound z y x
        let B = rbound (-1) ( 0) ( 0)
        let N = rbound ( 0) (-1) ( 0)
        let W = rbound ( 0) ( 0) (-1)
        let C = rbound ( 0) ( 0) ( 0)
        let E = rbound ( 0) ( 0) ( 1)
        let S = rbound ( 0) ( 1) ( 0)
        let T = rbound ( 1) ( 0) ( 0)
        in mean_7points (B,N,W,C,E,S,T)
        )

let single_iteration_stencil_7points arr =
  let ixs = [(-1,0,0),(0,-1,0),(0,0,-1),(0,0,0),(0,0,1),(0,1,0),(1,0,0)] in
  let f _ v = mean_7points (v[0],v[1],v[2],v[3],v[4],v[5],v[6]) in
  let empty = map (map (map (const ()))) arr in
  stencil_3d ixs f empty arr

let single_iteration_maps_13points [Nz][Ny][Nx] (arr:[Nz][Ny][Nx]f32) =
  let bound = edgeHandling.extendEdge3D arr (Nz-1) (Ny-1) (Nx-1)
  in tabulate_3d Nz Ny Nx (\z y x ->
        let rbound = bound z y x
        let b1 = rbound (-2) ( 0) ( 0)
        let b2 = rbound (-1) ( 0) ( 0)
        let n1 = rbound ( 0) (-2) ( 0)
        let n2 = rbound ( 0) (-1) ( 0)
        let w1 = rbound ( 0) ( 0) (-2)
        let w2 = rbound ( 0) ( 0) (-1)
        let c  = rbound ( 0) ( 0) ( 0)
        let e1 = rbound ( 0) ( 0) ( 1)
        let e2 = rbound ( 0) ( 0) ( 2)
        let s1 = rbound ( 0) ( 1) ( 0)
        let s2 = rbound ( 0) ( 2) ( 0)
        let t1 = rbound ( 1) ( 0) ( 0)
        let t2 = rbound ( 2) ( 0) ( 0)
        in mean_13points (b1,b2,n1,n2,w1,w2,c,e1,e2,s1,s2,t1,t2)
        )

let single_iteration_stencil_13points arr =
  let ixs = [(-2,0,0),(-1,0,0),(0,-2,0),(0,-1,0),(0,0,-2),(0,0,-1)
            ,(0,0,0)
            ,(0,0,1),(0,0,2),(0,1,0),(0,2,0),(1,0,0),(2,0,0)] in
  let f _ v = mean_13points (v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],v[10],v[11],v[12]) in
  let empty = map (map (map (const ()))) arr in
  stencil_3d ixs f empty arr

let num_iterations: i32 = 5
let compute_iters [Nz][Ny][Nx] f (arr:[Nz][Ny][Nx]f32)
  : [Nz][Ny][Nx]f32 =
  iterate num_iterations f arr

module jacobi3dCommon = {
  let bench_7p_maps     = compute_iters single_iteration_maps_7points
  let bench_7p_stencil  = compute_iters single_iteration_stencil_7points
  let bench_13p_maps    = compute_iters single_iteration_maps_13points
  let bench_13p_stencil = compute_iters single_iteration_stencil_13points
}
