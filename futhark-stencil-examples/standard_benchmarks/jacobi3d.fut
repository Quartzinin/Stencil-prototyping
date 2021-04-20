-- code is based on
-- https://gitlab.com/larisa.stoltzfus/liftstencil-cgo2018-artifact/-/blob/master/benchmarks/figure7/workflow1/reference/hotspot3D/hotspotKernel.cl
--
-- ==
-- entry: bench_7p_maps
-- random input { [256][256][256]f32 }
-- random input { [512][512][512]f32 }

-- ==
-- entry: bench_7p_stencil
-- random input { [256][256][256]f32 }
-- random input { [512][512][512]f32 }

-- ==
-- entry: bench_13p_maps
-- random input { [256][256][256]f32 }
-- random input { [512][512][512]f32 }

-- ==
-- entry: bench_13p_stencil
-- random input { [256][256][256]f32 }
-- random input { [512][512][512]f32 }

let mean_7points
    (p: (f32,f32,f32,f32,f32,f32,f32))
    : f32 = (p.0+p.1+p.2+p.3+p.4+p.5+p.6) / 7
let mean_13points
    (p: (f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32))
    : f32 = (p.0+p.1+p.2+p.3+p.4+p.5+p.6+p.7+p.8+p.9+p.10+p.11+p.12) / 13

let single_iteration_maps_7points [Nz][Ny][Nx] (arr:[Nz][Ny][Nx]f32) =
  let bound z y x =
    let bz = i64.min (i64.max 0 z) (Nz-1)
    let by = i64.min (i64.max 0 y) (Ny-1)
    let bx = i64.min (i64.max 0 x) (Nx-1)
    in arr[bz,by,bx]
  in tabulate_3d Nz Ny Nx (\z y x ->
        let b = bound (z-1) (y  ) (x  )
        let n = bound (z  ) (y-1) (x  )
        let w = bound (z  ) (y  ) (x-1)
        let c = bound (z  ) (y  ) (x  )
        let e = bound (z  ) (y  ) (x+1)
        let s = bound (z  ) (y+1) (x  )
        let t = bound (z+1) (y  ) (x  )
        in mean_7points (b,n,w,c,e,s,t)
        )

let single_iteration_stencil_7points arr =
  let ixs = [(-1,0,0),(0,-1,0),(0,0,-1),(0,0,0),(0,0,1),(0,1,0),(1,0,0)] in
  let f _ v = mean_7points (v[0],v[1],v[2],v[3],v[4],v[5],v[6]) in
  let empty = map (map (map (const ()))) arr in
  stencil_3d ixs f empty arr

let single_iteration_maps_13points [Nz][Ny][Nx] (arr:[Nz][Ny][Nx]f32) =
  let bound z y x =
    let bz = i64.min (i64.max 0 z) (Nz-1)
    let by = i64.min (i64.max 0 y) (Ny-1)
    let bx = i64.min (i64.max 0 x) (Nx-1)
    in arr[bz,by,bx]
  in tabulate_3d Nz Ny Nx (\z y x ->
        let b1 = bound (z-2) (y  ) (x  )
        let b2 = bound (z-1) (y  ) (x  )
        let n1 = bound (z  ) (y-2) (x  )
        let n2 = bound (z  ) (y-1) (x  )
        let w1 = bound (z  ) (y  ) (x-2)
        let w2 = bound (z  ) (y  ) (x-1)
        let c  = bound (z  ) (y  ) (x  )
        let e1 = bound (z  ) (y  ) (x+1)
        let e2 = bound (z  ) (y  ) (x+2)
        let s1 = bound (z  ) (y+1) (x  )
        let s2 = bound (z  ) (y+2) (x  )
        let t1 = bound (z+1) (y  ) (x  )
        let t2 = bound (z+2) (y  ) (x  )
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

entry bench_7p_maps     = compute_iters single_iteration_maps_7points
entry bench_7p_stencil  = compute_iters single_iteration_stencil_7points
entry bench_13p_maps    = compute_iters single_iteration_maps_13points
entry bench_13p_stencil = compute_iters single_iteration_stencil_13points
