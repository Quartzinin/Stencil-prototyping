--
-- ==
-- entry: bench_maps
-- random input { [256][256][256]f32 }
-- random input { [512][512][512]f32 }

-- ==
-- entry: bench_stencil
-- random input { [256][256][256]f32 }
-- random input { [512][512][512]f32 }


let updater B N W C E S T : f32
  = 0.125*(T - 2.0 * C + B)
  + 0.125*(S - 2.0 * C + N)
  + 0.125*(E - 2.0 * C + W)
  + C

let single_iteration_maps [len_z][len_y][len_x]
    (arr: [len_z][len_y][len_x]f32)
    : [len_z][len_y][len_x]f32 =
  let bound z y x =
    let bz = i64.min (i64.max 0 z) (len_z-1)
    let by = i64.min (i64.max 0 y) (len_y-1)
    let bx = i64.min (i64.max 0 x) (len_x-1)
    in arr[bz,by,bx]
  in tabulate_3d len_z len_y len_x (\z y x ->
        let B = bound (z-1) (y  ) (x  )
        let N = bound (z  ) (y-1) (x  )
        let W = bound (z  ) (y  ) (x-1)
        let C = bound (z  ) (y  ) (x  )
        let E = bound (z  ) (y  ) (x+1)
        let S = bound (z  ) (y+1) (x  )
        let T = bound (z+1) (y  ) (x  )
        in updater B N W C E S T
        )

let single_iteration_stencil [len_z][len_y][len_x]
    (arr: [len_z][len_y][len_x]f32)
    : [len_z][len_y][len_x]f32 =
  let ixs = [(-1,0,0),(0,-1,0),(0,0,-1),(0,0,0),(0,0,1),(0,1,0),(1,0,0)] in
  let f _ v = updater v[0] v[1] v[2] v[3] v[4] v[5] v[6] in
  let empty = map (map (map (const ()))) arr in
  stencil_3d ixs f empty arr

let num_iterations: i32 = 5
let compute_tran_temp [len_z][len_y][len_x]
  f (arr: [len_z][len_y][len_x]f32)
  : [len_z][len_y][len_x]f32 =
  iterate num_iterations f arr

entry bench_maps    = compute_tran_temp single_iteration_maps
entry bench_stencil = compute_tran_temp single_iteration_stencil
