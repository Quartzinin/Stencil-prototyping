-- code is based on
-- https://gitlab.com/larisa.stoltzfus/liftstencil-cgo2018-artifact/-/blob/master/benchmarks/figure7/workflow1/reference/hotspot3D/hotspotKernel.cl

let mean_5points
    (p: (f32,f32,f32,f32,f32))
    : f32 = (p.0+p.1+p.2+p.3+p.4) / 5
let mean_9points
    (p: (f32,f32,f32,f32,f32,f32,f32,f32,f32))
    : f32 = (p.0+p.1+p.2+p.3+p.4+p.5+p.6+p.7+p.8) / 9

let single_iteration_maps_5points [Ny][Nx] (arr:[Ny][Nx]f32) =
  let bound y x =
    let by = i64.min (i64.max 0 y) (Ny-1)
    let bx = i64.min (i64.max 0 x) (Nx-1)
    in #[unsafe] arr[by,bx]
  in tabulate_2d Ny Nx (\y x ->
        let n = bound (y-1) (x  )
        let w = bound (y  ) (x-1)
        let c = bound (y  ) (x  )
        let e = bound (y  ) (x+1)
        let s = bound (y+1) (x  )
        in mean_5points (n,w,c,e,s)
        )

let single_iteration_stencil_5points arr =
  let ixs = [(-1,0),(0,-1),(0,0),(0,1),(1,0)] in
  let f _ v = mean_5points (v[0],v[1],v[2],v[3],v[4]) in
  let empty = map (map (const ())) arr in
  stencil_2d ixs f empty arr

let single_iteration_maps_9points [Ny][Nx] (arr:[Ny][Nx]f32) =
  let bound y x =
    let by = i64.min (i64.max 0 y) (Ny-1)
    let bx = i64.min (i64.max 0 x) (Nx-1)
    in #[unsafe] arr[by,bx]
  in tabulate_2d Ny Nx (\y x ->
        let n1 = bound (y-2) (x  )
        let n2 = bound (y-1) (x  )
        let w1 = bound (y  ) (x-2)
        let w2 = bound (y  ) (x-1)
        let c  = bound (y  ) (x  )
        let e1 = bound (y  ) (x+1)
        let e2 = bound (y  ) (x+2)
        let s1 = bound (y+1) (x  )
        let s2 = bound (y+2) (x  )
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
