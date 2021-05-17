import "./edgeHandling"

let updater B N W C E S T : f32
  = 0.125*(T - 2.0 * C + B)
  + 0.125*(S - 2.0 * C + N)
  + 0.125*(E - 2.0 * C + W)
  + C

let single_iteration_maps [Nz][Ny][Nx]
    (arr: [Nz][Ny][Nx]f32)
    : [Nz][Ny][Nx]f32 =
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
        in updater B N W C E S T
        )

let single_iteration_stencil [Nz][Ny][Nx]
    (arr: [Nz][Ny][Nx]f32)
    : [Nz][Ny][Nx]f32 =
  let ixs = [(-1,0,0),(0,-1,0),(0,0,-1),(0,0,0),(0,0,1),(0,1,0),(1,0,0)] in
  let f _ v = updater v[0] v[1] v[2] v[3] v[4] v[5] v[6] in
  let empty = map (map (map (const ()))) arr in
  stencil_3d ixs f empty arr

let num_iterations: i32 = 5
let compute_tran_temp [Nz][Ny][Nx]
  f (arr: [Nz][Ny][Nx]f32)
  : [Nz][Ny][Nx]f32 =
  iterate num_iterations f arr

module heat3dCommon = {
    let bench_maps    = compute_tran_temp single_iteration_maps
    let bench_stencil = compute_tran_temp single_iteration_stencil
}
