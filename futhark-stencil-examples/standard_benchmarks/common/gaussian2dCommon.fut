import "./edgeHandling"

let sigma:f32 = 1.5
let gaussian_2d x y = f32.e**(-((x**2 + y**2)/(2*sigma**2))) * (1/(2*f32.pi*sigma**2))
-- weights are based on absolute distances to centerpoint
let raw_weights = ( gaussian_2d 0 0, gaussian_2d 1 0, gaussian_2d 1 1
                  , gaussian_2d 0 2, gaussian_2d 1 2, gaussian_2d 2 2)
-- number of instances of each weight times the weight.
let weight_sum =
        4*raw_weights.5
      + 8*raw_weights.4
      + 4*raw_weights.3
      + 4*raw_weights.2
      + 4*raw_weights.1
      + 1*raw_weights.0
-- normalizied weights.
let nrmw =
    ( raw_weights.0 / weight_sum
    , raw_weights.1 / weight_sum
    , raw_weights.2 / weight_sum
    , raw_weights.3 / weight_sum
    , raw_weights.4 / weight_sum
    , raw_weights.5 / weight_sum
    )

-- gaussian weighted mean / blur
let gauss_blur
    (p: (f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32))
    : f32
--    = p.0 *nrmw.5 + p.1 *nrmw.4 + p.2 *nrmw.3 + p.3 *nrmw.4 + p.4 *nrmw.5
--    + p.5 *nrmw.4 + p.6 *nrmw.2 + p.7 *nrmw.1 + p.8 *nrmw.2 + p.9 *nrmw.4
--    + p.11*nrmw.3 + p.12*nrmw.1 + p.13*nrmw.0 + p.14*nrmw.1 + p.15*nrmw.3
--    + p.15*nrmw.4 + p.16*nrmw.2 + p.17*nrmw.1 + p.18*nrmw.2 + p.19*nrmw.2
--    + p.20*nrmw.5 + p.21*nrmw.4 + p.22*nrmw.3 + p.23*nrmw.4 + p.24*nrmw.1
-- using distribution of multiplication over addition.
    = nrmw.0 * (p.13)
    + nrmw.1 * (p.7+p.12+p.14+p.17)
    + nrmw.2 * (p.6+p.8+p.16+p.18)
    + nrmw.3 * (p.2+p.11+p.15+p.22)
    + nrmw.4 * (p.1+p.3+p.5+p.9+p.15+p.19+p.21+p.23)
    + nrmw.5 * (p.0+p.4+p.20+p.24)


let single_iteration_maps_25points [Ny][Nx] fun (arr:[Ny][Nx]f32) =
  let bound = edgeHandling.extendEdge2D arr (Ny-1) (Nx-1)
  in tabulate_2d Ny Nx (\y x ->
        let rbound = bound y x
        let n1  = rbound (-2) (-2)
        let n2  = rbound (-2) (-1)
        let n3  = rbound (-2) ( 0)
        let n4  = rbound (-2) ( 1)
        let n5  = rbound (-2) ( 2)
        let n6  = rbound (-1) (-2)
        let n7  = rbound (-1) (-1)
        let n8  = rbound (-1) ( 0)
        let n9  = rbound (-1) ( 1)
        let n10 = rbound (-1) ( 2)
        let n11 = rbound ( 0) (-2)
        let n12 = rbound ( 0) (-1)
        let n13 = rbound ( 0) ( 0)
        let n14 = rbound ( 0) ( 1)
        let n15 = rbound ( 0) ( 2)
        let n16 = rbound ( 1) (-2)
        let n17 = rbound ( 1) (-1)
        let n18 = rbound ( 1) ( 0)
        let n19 = rbound ( 1) ( 1)
        let n20 = rbound ( 1) ( 2)
        let n21 = rbound ( 2) (-2)
        let n22 = rbound ( 2) (-1)
        let n23 = rbound ( 2) ( 0)
        let n24 = rbound ( 2) ( 1)
        let n25 = rbound ( 2) ( 2)
        in fun (n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,n16,n17,n18,n19,n20,n21,n22,n23,n24,n25)
        )

let single_iteration_stencil_25points fun arr =
  let ixs = [(-2,-2),(-2,-1),(-2, 0),(-2, 1),(-2, 2)
            ,(-1,-2),(-1,-1),(-1, 0),(-1, 1),(-1, 2)
            ,( 0,-2),( 0,-1),( 0, 0),( 0, 1),( 0, 2)
            ,( 1,-2),( 1,-1),( 1, 0),( 1, 1),( 1, 2)
            ,( 2,-2),( 2,-1),( 2, 0),( 2, 1),( 2, 2)] in
  let f _ v = fun (v[0] ,v[1] ,v[2] ,v[3] ,v[4]
                  ,v[5] ,v[6] ,v[7] ,v[8] ,v[9]
                  ,v[10],v[11],v[12],v[13],v[14]
                  ,v[15],v[16],v[17],v[18],v[19]
                  ,v[20],v[21],v[22],v[23],v[24]) in
  let empty = map (map (const ())) arr in
  stencil_2d ixs f empty arr

let num_iterations: i32 = 5

let compute_iters f arr = iterate num_iterations (f gauss_blur) arr

module gaussian2dCommon = {
  let bench_25p_maps    = compute_iters single_iteration_maps_25points
  let bench_25p_stencil = compute_iters single_iteration_stencil_25points
}
