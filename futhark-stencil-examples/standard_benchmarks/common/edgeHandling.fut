

let boundc (s:i64) (m:i64) =
  let tmp = if s < 0 then 0 else s in if tmp > m then m else tmp
let boundm (s:i64) (r:i64) (m:i64) =
  let rx = r + s in
  if r < 0      then i64.max 0 rx
  else if r > 0 then i64.min m rx
  else               rx

module edgeHandling = {

--  let extendEdge2D arr (maxy:i64) (maxx:i64) (y:i64) (x:i64) =
--    let by = boundc y maxy
--    let bx = boundc x maxx
--    in #[unsafe] arr[by,bx]
--  let extendEdge3D arr (maxz:i64) (maxy:i64) (maxx:i64) (z:i64) (y:i64) (x:i64) =
--    let bz = boundc z maxz
--    let by = boundc y maxy
--    let bx = boundc x maxx
--    in #[unsafe] arr[bz,by,bx]

 let extendEdge2D arr (maxy:i64) (maxx:i64) (y:i64) (x:i64) (ry:i64) (rx:i64) =
    let by = boundm y ry maxy
    let bx = boundm x rx maxx
    in #[unsafe] arr[by,bx]
 let extendEdge3D arr (maxz:i64) (maxy:i64) (maxx:i64) (z:i64) (y:i64) (x:i64) (rz:i64) (ry:i64) (rx:i64) =
    let bz = boundm z rz maxz
    let by = boundm y ry maxy
    let bx = boundm x rx maxx
    in #[unsafe] arr[bz,by,bx]
}
