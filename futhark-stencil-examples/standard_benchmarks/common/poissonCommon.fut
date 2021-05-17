import "./edgeHandling"

let updater
    (C:f32
    ,(ed:(f32,f32,f32,f32,f32,f32))
    ,(edd:(f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32))
    ) : f32
  = 2.6666 * C
  - 0.1666 * (ed.0+ed.1+ed.2+ed.3+ed.4+ed.5)
  - 0.0833 * (edd.0+edd.1+edd.2+edd.3+edd.4+edd.5+edd.6+edd.7+edd.8+edd.9+edd.10+edd.11)

let single_iteration_maps [Nz][Ny][Nx]
    (arr: [Nz][Ny][Nx]f32)
    : [Nz][Ny][Nx]f32 =
  let bound = edgeHandling.extendEdge3D arr (Nz-1) (Ny-1) (Nx-1)
  in tabulate_3d Nz Ny Nx (\z y x ->
        let rbound = bound z y x
        let e7  = rbound (-1) (-1) ( 0)
        let e11 = rbound (-1) ( 0) (-1)
        let e1  = rbound (-1) ( 0) ( 0)
        let e15 = rbound (-1) ( 0) ( 1)
        let e9  = rbound (-1) ( 1) ( 0)
        let e13 = rbound ( 0) (-1) (-1)
        let e3  = rbound ( 0) (-1) ( 0)
        let e17 = rbound ( 0) (-1) ( 1)
        let e5  = rbound ( 0) ( 0) (-1)
        let e19 = rbound ( 0) ( 0) ( 0)
        let e6  = rbound ( 0) ( 0) ( 1)
        let e14 = rbound ( 0) ( 1) (-1)
        let e4  = rbound ( 0) ( 1) ( 0)
        let e18 = rbound ( 0) ( 1) ( 1)
        let e8  = rbound ( 1) (-1) ( 0)
        let e12 = rbound ( 1) ( 0) (-1)
        let e2  = rbound ( 1) ( 0) ( 0)
        let e16 = rbound ( 1) ( 0) ( 1)
        let e10 = rbound ( 1) ( 1) ( 0)
        in updater (e19,(e1,e2,e3,e4,e5,e6),(e7,e8,e9,e10,e11,e12,e13,e14,e15,e16,e17,e18))
        )

let single_iteration_stencil [Nz][Ny][Nx]
    (arr: [Nz][Ny][Nx]f32)
    : [Nz][Ny][Nx]f32 =
  let ixs = [(-1,-1,0),(-1,0,-1),(-1,0,0),(-1,0,1),(-1,1,0)
            ,(0,-1,-1),(0,-1,0),(0,-1,1),(0,0,-1),(0,0,0),(0,0,1),(0,1,-1),(0,1,0),(0,1,1)
            ,(1,-1,0),(1,0,-1),(1,0,0),(1,0,1),(1,1,0)] in
  let f _ v = updater (v[9]
                      ,(v[2],v[6],v[8],v[10],v[12],v[17])
                      ,(v[0],v[1],v[3],v[4],v[5],v[7],v[11],v[13],v[14],v[15],v[16],v[18])) in
  let empty = map (map (map (const ()))) arr in
  stencil_3d ixs f empty arr

let num_iterations: i32 = 5
let compute_iters [Nz][Ny][Nx]
  f (arr: [Nz][Ny][Nx]f32)
  : [Nz][Ny][Nx]f32 =
  iterate num_iterations f arr

module poissonCommon = {
  let bench_maps    = compute_iters single_iteration_maps
  let bench_stencil = compute_iters single_iteration_stencil
}
