-- ==
-- entry: bench_5p_maps
-- random input { [4096][4096]f32 }
-- random input { [8192][8192]f32 }

-- ==
-- entry: bench_5p_stencil
-- random input { [4096][4096]f32 }
-- random input { [8192][8192]f32 }

let avoid_div_zero: f32 = 0.0001
let gradiant_5point ((N,W,C,E,S): (f32,f32,f32,f32,f32)) : f32 =
    let ns = let tmp = C-N in tmp*tmp in
    let ws = let tmp = C-W in tmp*tmp in
    let es = let tmp = C-E in tmp*tmp in
    let ss = let tmp = C-S in tmp*tmp in
    C + (1.0 / f32.sqrt (avoid_div_zero + ns + ws + es + ss))

let single_iteration_maps_5points [Ny][Nx] (arr:[Ny][Nx]f32) =
  let bound y x =
    let by = i64.min (i64.max 0 y) (Ny-1)
    let bx = i64.min (i64.max 0 x) (Nx-1)
    in arr[by,bx]
  in tabulate_2d Ny Nx (\y x ->
        let N = bound (y-1) (x  )
        let W = bound (y  ) (x-1)
        let C = bound (y  ) (x  )
        let E = bound (y  ) (x+1)
        let S = bound (y+1) (x  )
        in gradiant_5point (N,W,C,E,S)
        )

let single_iteration_stencil_5points arr =
  let ixs = [(-1, 0),( 0,-1),( 0, 0),( 0, 1),( 1, 0)] in
  let f _ v = gradiant_5point (v[0] ,v[1] ,v[2] ,v[3] ,v[4]) in
  let empty = map (map (const ())) arr in
  stencil_2d ixs f empty arr

entry bench_5p_maps    = single_iteration_maps_5points
entry bench_5p_stencil = single_iteration_stencil_5points
