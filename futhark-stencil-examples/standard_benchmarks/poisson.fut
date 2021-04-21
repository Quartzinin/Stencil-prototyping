--
-- ==
-- entry: bench_maps
-- random input { [256][256][256]f32 }
-- random input { [512][512][512]f32 }

-- ==
-- entry: bench_stencil
-- random input { [256][256][256]f32 }
-- random input { [512][512][512]f32 }


let updater
    (C:f32
    ,(ed:(f32,f32,f32,f32,f32,f32))
    ,(edd:(f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32))
    ) : f32
  = 2.6666 * C
  - 0.1666 * (ed.0+ed.1+ed.2+ed.3+ed.4+ed.5)
  - 0.0833 * (edd.0+edd.1+edd.2+edd.3+edd.4+edd.5+edd.6+edd.7+edd.8+edd.9+edd.10+edd.11)

let single_iteration_maps [len_z][len_y][len_x]
    (arr: [len_z][len_y][len_x]f32)
    : [len_z][len_y][len_x]f32 =
  let bound z y x =
    let bz = i64.min (i64.max 0 z) (len_z-1)
    let by = i64.min (i64.max 0 y) (len_y-1)
    let bx = i64.min (i64.max 0 x) (len_x-1)
    in arr[bz,by,bx]
  in tabulate_3d len_z len_y len_x (\z y x ->
        let e7  = bound (z-1) (y-1) (x  )
        let e11 = bound (z-1) (y  ) (x-1)
        let e1  = bound (z-1) (y  ) (x  )
        let e15 = bound (z-1) (y  ) (x+1)
        let e9  = bound (z-1) (y+1) (x  )
        let e13 = bound (z  ) (y-1) (x-1)
        let e3  = bound (z  ) (y-1) (x  )
        let e17 = bound (z  ) (y-1) (x+1)
        let e5  = bound (z  ) (y  ) (x-1)
        let e19 = bound (z  ) (y  ) (x  )
        let e6  = bound (z  ) (y  ) (x+1)
        let e14 = bound (z  ) (y+1) (x-1)
        let e4  = bound (z  ) (y+1) (x  )
        let e18 = bound (z  ) (y+1) (x+1)
        let e8  = bound (z+1) (y-1) (x  )
        let e12 = bound (z+1) (y  ) (x-1)
        let e2  = bound (z+1) (y  ) (x  )
        let e16 = bound (z+1) (y  ) (x+1)
        let e10 = bound (z+1) (y+1) (x  )
        in updater (e19,(e1,e2,e3,e4,e5,e6),(e7,e8,e9,e10,e11,e12,e13,e14,e15,e16,e17,e18))
        )

let single_iteration_stencil [len_z][len_y][len_x]
    (arr: [len_z][len_y][len_x]f32)
    : [len_z][len_y][len_x]f32 =
  let ixs = [(-1,-1,0),(-1,0,-1),(-1,0,0),(-1,0,1),(-1,1,0)
            ,(0,-1,-1),(0,-1,0),(0,-1,1),(0,0,-1),(0,0,0),(0,0,1),(0,1,-1),(0,1,0),(0,1,1)
            ,(1,-1,0),(1,0,-1),(1,0,0),(1,0,1),(1,1,0)] in
  let f _ v = updater (v[9]
                      ,(v[2],v[6],v[8],v[10],v[12],v[17])
                      ,(v[0],v[1],v[3],v[4],v[5],v[7],v[11],v[13],v[14],v[15],v[16],v[18])) in
  let empty = map (map (map (const ()))) arr in
  stencil_3d ixs f empty arr

let num_iterations: i32 = 5
let compute_iters [len_z][len_y][len_x]
  f (arr: [len_z][len_y][len_x]f32)
  : [len_z][len_y][len_x]f32 =
  iterate num_iterations f arr

entry bench_maps    = compute_iters single_iteration_maps
entry bench_stencil = compute_iters single_iteration_stencil
