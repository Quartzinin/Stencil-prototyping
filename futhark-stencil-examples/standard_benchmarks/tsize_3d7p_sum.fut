import "./common/edgeHandling"

let stencil_fun 'a
      (p:a->a->a) (x1:a) (x2:a) (x3:a) (x4:a) (x5:a) (x6:a) (x7:a)
      : a
      = p (p (p (p (p (p x1 x2) x3) x4) x5) x6) x7

let nest_maps 'a [Nz][Ny][Nx] (p:a->a->a) (inp:[Nz][Ny][Nx]a) =
  let fun = stencil_fun p
  let siter arr =
    let bound = edgeHandling.extendEdge3D arr (Nz-1) (Ny-1) (Nx-1) in
    tabulate_3d Nz Ny Nx (\z y x ->
      let rbound = bound z y x
      let b = rbound (-1) ( 0) ( 0)
      let n = rbound ( 0) (-1) ( 0)
      let w = rbound ( 0) ( 0) (-1)
      let c = rbound ( 0) ( 0) ( 0)
      let e = rbound ( 0) ( 0) ( 1)
      let s = rbound ( 0) ( 1) ( 0)
      let t = rbound ( 1) ( 0) ( 0)
      in fun b n w c e s t
      )
  in siter inp

let stencil p inp =
  let ixs = [(-1,0,0),(0,-1,0),(0,0,-1),(0,0,0),(0,0,1),(0,1,0),(1,0,0)] in
  let f c v = stencil_fun p v[0] v[1] v[2] c v[4] v[5] v[6] in
  stencil_3d ixs f inp inp

-- highlights a bug.
-- let stencil2d p inp = stencil_2d [(-1,0),(1,0)] (\c v -> p (p c v[0]) v[1]) inp inp
-- 1==
-- entry: main_stencil2d_i8
-- compiled random input { [255][255]i8 } auto output
--entry main_stencil2d_i8 = stencil2d (i8.+)


-- ==
-- entry: main_stencil_i8
-- compiled random input { [255][255][255]i8 } auto output
entry main_stencil_i8 = stencil (i8.+)

-- ==
-- entry: main_nest_maps_i8
-- compiled random input { [255][255][255]i8 } auto output
entry main_nest_maps_i8 = nest_maps (i8.+)

-- ==
-- entry: main_stencil_f64
-- compiled random input { [255][255][255]f64 } auto output
entry main_stencil_f64 = stencil (f64.+)

-- ==
-- entry: main_nest_maps_f64
-- compiled random input { [255][255][255]f64 } auto output
entry main_nest_maps_f64 = nest_maps (f64.+)
