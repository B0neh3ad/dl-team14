from dsl import *
from constants import *


def solve_67a3c6ac(I):
    O = vmirror(I)
    return O


def solve_68b16354(I):
    O = hmirror(I)
    return O


def solve_74dd1130(I):
    O = dmirror(I)
    return O


def solve_3c9b0459(I):
    O = rot180(I)
    return O


def solve_6150a2bd(I):
    O = rot180(I)
    return O


def solve_9172f3a0(I):
    O = upscale(I, THREE)
    return O


def solve_9dfd6313(I):
    O = dmirror(I)
    return O


def solve_a416b8f3(I):
    O = hconcat(I, I)
    return O


def solve_b1948b0a(I):
    O = replace(I, SIX, TWO)
    return O


def solve_c59eb873(I):
    O = upscale(I, TWO)
    return O


def solve_c8f0f002(I):
    O = replace(I, SEVEN, FIVE)
    return O


def solve_d10ecb37(I):
    O = crop(I, ORIGIN, TWO_BY_TWO)
    return O


def solve_d511f180(I):
    O = switch(I, FIVE, EIGHT)
    return O


def solve_ed36ccf7(I):
    O = rot270(I)
    return O


def solve_4c4377d9(I):
    x1 = hmirror(I)
    O = vconcat(x1, I)
    return O


def solve_6d0aefbc(I):
    x1 = vmirror(I)
    O = hconcat(I, x1)
    return O


def solve_6fa7a44f(I):
    x1 = hmirror(I)
    O = vconcat(I, x1)
    return O


def solve_5614dbcf(I):
    x1 = replace(I, FIVE, ZERO)
    O = downscale(x1, THREE)
    return O


def solve_5bd6f4ac(I):
    x1 = tojvec(SIX)
    O = crop(I, x1, THREE_BY_THREE)
    return O


def solve_5582e5ca(I):
    x1 = mostcolor(I)
    O = canvas(x1, THREE_BY_THREE)
    return O


def solve_8be77c9e(I):
    x1 = hmirror(I)
    O = vconcat(I, x1)
    return O


def solve_c9e6f938(I):
    x1 = vmirror(I)
    O = hconcat(I, x1)
    return O


def solve_2dee498d(I):
    x1 = hsplit(I, THREE)
    O = first(x1)
    return O


def solve_1cf80156(I):
    x1 = objects(I, T, T, T)
    x2 = first(x1)
    O = subgrid(x2, I)
    return O


def solve_32597951(I):
    x1 = ofcolor(I, EIGHT)
    x2 = delta(x1)
    O = fill(I, THREE, x2)
    return O


def solve_25ff71a9(I):
    x1 = objects(I, T, T, T)
    x2 = first(x1)
    O = move(I, x2, DOWN)
    return O


def solve_0b148d64(I):
    x1 = partition(I)
    x2 = argmin(x1, size)
    O = subgrid(x2, I)
    return O


def solve_1f85a75f(I):
    x1 = objects(I, T, T, T)
    x2 = argmax(x1, size)
    O = subgrid(x2, I)
    return O


def solve_23b5c85d(I):
    x1 = objects(I, T, T, T)
    x2 = argmin(x1, size)
    O = subgrid(x2, I)
    return O


def solve_9ecd008a(I):
    x1 = vmirror(I)
    x2 = ofcolor(I, ZERO)
    O = subgrid(x2, x1)
    return O


def solve_ac0a08a4(I):
    x1 = colorcount(I, ZERO)
    x2 = subtract(NINE, x1)
    O = upscale(I, x2)
    return O


def solve_be94b721(I):
    x1 = objects(I, T, F, T)
    x2 = argmax(x1, size)
    O = subgrid(x2, I)
    return O


def solve_c909285e(I):
    x1 = leastcolor(I)
    x2 = ofcolor(I, x1)
    O = subgrid(x2, I)
    return O


def solve_f25ffba3(I):
    x1 = bottomhalf(I)
    x2 = hmirror(x1)
    O = vconcat(x2, x1)
    return O


def solve_c1d99e64(I):
    x1 = frontiers(I)
    x2 = merge(x1)
    O = fill(I, TWO, x2)
    return O


def solve_b91ae062(I):
    x1 = numcolors(I)
    x2 = decrement(x1)
    O = upscale(I, x2)
    return O


def solve_3aa6fb7a(I):
    x1 = objects(I, T, F, T)
    x2 = mapply(corners, x1)
    O = underfill(I, ONE, x2)
    return O


def solve_7b7f7511(I):
    x1 = portrait(I)
    x2 = branch(x1, tophalf, lefthalf)
    O = x2(I)
    return O


def solve_4258a5f9(I):
    x1 = ofcolor(I, FIVE)
    x2 = mapply(neighbors, x1)
    O = fill(I, ONE, x2)
    return O


def solve_2dc579da(I):
    x1 = vsplit(I, TWO)
    x2 = rbind(hsplit, TWO)
    x3 = mapply(x2, x1)
    O = argmax(x3, numcolors)
    return O


def solve_28bf18c6(I):
    x1 = objects(I, T, T, T)
    x2 = first(x1)
    x3 = subgrid(x2, I)
    O = hconcat(x3, x3)
    return O


def solve_3af2c5a8(I):
    x1 = vmirror(I)
    x2 = hconcat(I, x1)
    x3 = hmirror(x2)
    O = vconcat(x2, x3)
    return O


def solve_44f52bb0(I):
    x1 = vmirror(I)
    x2 = equality(x1, I)
    x3 = branch(x2, ONE, SEVEN)
    O = canvas(x3, UNITY)
    return O


def solve_62c24649(I):
    x1 = vmirror(I)
    x2 = hconcat(I, x1)
    x3 = hmirror(x2)
    O = vconcat(x2, x3)
    return O


def solve_67e8384a(I):
    x1 = vmirror(I)
    x2 = hconcat(I, x1)
    x3 = hmirror(x2)
    O = vconcat(x2, x3)
    return O


def solve_7468f01a(I):
    x1 = objects(I, F, T, T)
    x2 = first(x1)
    x3 = subgrid(x2, I)
    O = vmirror(x3)
    return O


def solve_662c240a(I):
    x1 = vsplit(I, THREE)
    x2 = fork(equality, dmirror, identity)
    x3 = compose(flip, x2)
    O = extract(x1, x3)
    return O


def solve_42a50994(I):
    x1 = objects(I, T, T, T)
    x2 = sizefilter(x1, ONE)
    x3 = merge(x2)
    O = cover(I, x3)
    return O


def solve_56ff96f3(I):
    x1 = fgpartition(I)
    x2 = fork(recolor, color, backdrop)
    x3 = mapply(x2, x1)
    O = paint(I, x3)
    return O


def solve_50cb2852(I):
    x1 = objects(I, T, F, T)
    x2 = compose(backdrop, inbox)
    x3 = mapply(x2, x1)
    O = fill(I, EIGHT, x3)
    return O


def solve_4347f46a(I):
    x1 = objects(I, T, F, T)
    x2 = fork(difference, toindices, box)
    x3 = mapply(x2, x1)
    O = fill(I, ZERO, x3)
    return O


def solve_46f33fce(I):
    x1 = rot180(I)
    x2 = downscale(x1, TWO)
    x3 = rot180(x2)
    O = upscale(x3, FOUR)
    return O


def solve_a740d043(I):
    x1 = objects(I, T, T, T)
    x2 = merge(x1)
    x3 = subgrid(x2, I)
    O = replace(x3, ONE, ZERO)
    return O


def solve_a79310a0(I):
    x1 = objects(I, T, F, T)
    x2 = first(x1)
    x3 = move(I, x2, DOWN)
    O = replace(x3, EIGHT, TWO)
    return O


def solve_aabf363d(I):
    x1 = leastcolor(I)
    x2 = replace(I, x1, ZERO)
    x3 = leastcolor(x2)
    O = replace(x2, x3, x1)
    return O


def solve_ae4f1146(I):
    x1 = objects(I, F, F, T)
    x2 = rbind(colorcount, ONE)
    x3 = argmax(x1, x2)
    O = subgrid(x3, I)
    return O


def solve_b27ca6d3(I):
    x1 = objects(I, T, F, T)
    x2 = sizefilter(x1, TWO)
    x3 = mapply(outbox, x2)
    O = fill(I, THREE, x3)
    return O


def solve_ce22a75a(I):
    x1 = objects(I, T, F, T)
    x2 = apply(outbox, x1)
    x3 = mapply(backdrop, x2)
    O = fill(I, ONE, x3)
    return O


def solve_dc1df850(I):
    x1 = objects(I, T, F, T)
    x2 = colorfilter(x1, TWO)
    x3 = mapply(outbox, x2)
    O = fill(I, ONE, x3)
    return O


def solve_f25fbde4(I):
    x1 = objects(I, T, T, T)
    x2 = first(x1)
    x3 = subgrid(x2, I)
    O = upscale(x3, TWO)
    return O


def solve_44d8ac46(I):
    x1 = objects(I, T, F, T)
    x2 = apply(delta, x1)
    x3 = mfilter(x2, square)
    O = fill(I, TWO, x3)
    return O


def solve_1e0a9b12(I):
    x1 = rot270(I)
    x2 = rbind(order, identity)
    x3 = apply(x2, x1)
    O = rot90(x3)
    return O


def solve_0d3d703e(I):
    x1 = switch(I, THREE, FOUR)
    x2 = switch(x1, EIGHT, NINE)
    x3 = switch(x2, TWO, SIX)
    O = switch(x3, ONE, FIVE)
    return O


def solve_3618c87e(I):
    x1 = objects(I, T, F, T)
    x2 = sizefilter(x1, ONE)
    x3 = merge(x2)
    O = move(I, x3, TWO_BY_ZERO)
    return O


def solve_1c786137(I):
    x1 = objects(I, T, F, F)
    x2 = argmax(x1, height)
    x3 = subgrid(x2, I)
    O = trim(x3)
    return O


def solve_8efcae92(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, ONE)
    x3 = compose(size, delta)
    x4 = argmax(x2, x3)
    O = subgrid(x4, I)
    return O


def solve_445eab21(I):
    x1 = objects(I, T, F, T)
    x2 = fork(multiply, height, width)
    x3 = argmax(x1, x2)
    x4 = color(x3)
    O = canvas(x4, TWO_BY_TWO)
    return O


def solve_6f8cd79b(I):
    x1 = asindices(I)
    x2 = apply(initset, x1)
    x3 = rbind(bordering, I)
    x4 = mfilter(x2, x3)
    O = fill(I, EIGHT, x4)
    return O


def solve_2013d3e2(I):
    x1 = objects(I, F, T, T)
    x2 = first(x1)
    x3 = subgrid(x2, I)
    x4 = lefthalf(x3)
    O = tophalf(x4)
    return O


def solve_41e4d17e(I):
    x1 = objects(I, T, F, T)
    x2 = fork(combine, vfrontier, hfrontier)
    x3 = compose(x2, center)
    x4 = mapply(x3, x1)
    O = underfill(I, SIX, x4)
    return O


def solve_9565186b(I):
    x1 = shape(I)
    x2 = objects(I, T, F, F)
    x3 = argmax(x2, size)
    x4 = canvas(FIVE, x1)
    O = paint(x4, x3)
    return O


def solve_aedd82e4(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, TWO)
    x3 = sizefilter(x2, ONE)
    x4 = merge(x3)
    O = fill(I, ONE, x4)
    return O


def solve_bb43febb(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, FIVE)
    x3 = compose(backdrop, inbox)
    x4 = mapply(x3, x2)
    O = fill(I, TWO, x4)
    return O


def solve_e98196ab(I):
    x1 = tophalf(I)
    x2 = bottomhalf(I)
    x3 = objects(x1, T, F, T)
    x4 = merge(x3)
    O = paint(x2, x4)
    return O


def solve_f76d97a5(I):
    x1 = palette(I)
    x2 = first(x1)
    x3 = last(x1)
    x4 = switch(I, x2, x3)
    O = replace(x4, FIVE, ZERO)
    return O


def solve_ce9e57f2(I):
    x1 = objects(I, T, F, T)
    x2 = fork(connect, ulcorner, centerofmass)
    x3 = mapply(x2, x1)
    x4 = fill(I, EIGHT, x3)
    O = switch(x4, EIGHT, TWO)
    return O


def solve_22eb0ac0(I):
    x1 = fgpartition(I)
    x2 = fork(recolor, color, backdrop)
    x3 = apply(x2, x1)
    x4 = mfilter(x3, hline)
    O = paint(I, x4)
    return O


def solve_9f236235(I):
    x1 = compress(I)
    x2 = objects(I, T, F, F)
    x3 = vmirror(x1)
    x4 = valmin(x2, width)
    O = downscale(x3, x4)
    return O


def solve_a699fb00(I):
    x1 = ofcolor(I, ONE)
    x2 = shift(x1, RIGHT)
    x3 = shift(x1, LEFT)
    x4 = intersection(x2, x3)
    O = fill(I, TWO, x4)
    return O


def solve_46442a0e(I):
    x1 = rot90(I)
    x2 = rot180(I)
    x3 = rot270(I)
    x4 = hconcat(I, x1)
    x5 = hconcat(x3, x2)
    O = vconcat(x4, x5)
    return O


def solve_7fe24cdd(I):
    x1 = rot90(I)
    x2 = rot180(I)
    x3 = rot270(I)
    x4 = hconcat(I, x1)
    x5 = hconcat(x3, x2)
    O = vconcat(x4, x5)
    return O


def solve_0ca9ddb6(I):
    x1 = ofcolor(I, ONE)
    x2 = ofcolor(I, TWO)
    x3 = mapply(dneighbors, x1)
    x4 = mapply(ineighbors, x2)
    x5 = fill(I, SEVEN, x3)
    O = fill(x5, FOUR, x4)
    return O


def solve_543a7ed5(I):
    x1 = objects(I, T, F, T)
    x2 = colorfilter(x1, SIX)
    x3 = mapply(outbox, x2)
    x4 = fill(I, THREE, x3)
    x5 = mapply(delta, x2)
    O = fill(x4, FOUR, x5)
    return O


def solve_0520fde7(I):
    x1 = vmirror(I)
    x2 = lefthalf(x1)
    x3 = righthalf(x1)
    x4 = vmirror(x3)
    x5 = cellwise(x2, x4, ZERO)
    O = replace(x5, ONE, TWO)
    return O


def solve_dae9d2b5(I):
    x1 = lefthalf(I)
    x2 = righthalf(I)
    x3 = ofcolor(x1, FOUR)
    x4 = ofcolor(x2, THREE)
    x5 = combine(x3, x4)
    O = fill(x1, SIX, x5)
    return O


def solve_8d5021e8(I):
    x1 = vmirror(I)
    x2 = hconcat(x1, I)
    x3 = hmirror(x2)
    x4 = vconcat(x2, x3)
    x5 = vconcat(x4, x2)
    O = hmirror(x5)
    return O


def solve_928ad970(I):
    x1 = ofcolor(I, FIVE)
    x2 = subgrid(x1, I)
    x3 = trim(x2)
    x4 = leastcolor(x3)
    x5 = inbox(x1)
    O = fill(I, x4, x5)
    return O


def solve_b60334d2(I):
    x1 = ofcolor(I, FIVE)
    x2 = replace(I, FIVE, ZERO)
    x3 = mapply(dneighbors, x1)
    x4 = mapply(ineighbors, x1)
    x5 = fill(x2, ONE, x3)
    O = fill(x5, FIVE, x4)
    return O


def solve_b94a9452(I):
    x1 = objects(I, F, F, T)
    x2 = first(x1)
    x3 = subgrid(x2, I)
    x4 = leastcolor(x3)
    x5 = mostcolor(x3)
    O = switch(x3, x4, x5)
    return O


def solve_d037b0a7(I):
    x1 = objects(I, T, F, T)
    x2 = rbind(shoot, DOWN)
    x3 = compose(x2, center)
    x4 = fork(recolor, color, x3)
    x5 = mapply(x4, x1)
    O = paint(I, x5)
    return O


def solve_d0f5fe59(I):
    x1 = objects(I, T, F, T)
    x2 = size(x1)
    x3 = astuple(x2, x2)
    x4 = canvas(ZERO, x3)
    x5 = shoot(ORIGIN, UNITY)
    O = fill(x4, EIGHT, x5)
    return O


def solve_e3497940(I):
    x1 = lefthalf(I)
    x2 = righthalf(I)
    x3 = vmirror(x2)
    x4 = objects(x3, T, F, T)
    x5 = merge(x4)
    O = paint(x1, x5)
    return O


def solve_e9afcf9a(I):
    x1 = astuple(TWO, ONE)
    x2 = crop(I, ORIGIN, x1)
    x3 = hmirror(x2)
    x4 = hconcat(x2, x3)
    x5 = hconcat(x4, x4)
    O = hconcat(x5, x4)
    return O


def solve_48d8fb45(I):
    x1 = objects(I, T, T, T)
    x2 = matcher(size, ONE)
    x3 = extract(x1, x2)
    x4 = lbind(adjacent, x3)
    x5 = extract(x1, x4)
    O = subgrid(x5, I)
    return O


def solve_d406998b(I):
    x1 = vmirror(I)
    x2 = ofcolor(x1, FIVE)
    x3 = compose(even, last)
    x4 = sfilter(x2, x3)
    x5 = fill(x1, THREE, x4)
    O = vmirror(x5)
    return O


def solve_5117e062(I):
    x1 = objects(I, F, T, T)
    x2 = matcher(numcolors, TWO)
    x3 = extract(x1, x2)
    x4 = subgrid(x3, I)
    x5 = mostcolor(x3)
    O = replace(x4, EIGHT, x5)
    return O


def solve_3906de3d(I):
    x1 = rot270(I)
    x2 = rbind(order, identity)
    x3 = switch(x1, ONE, TWO)
    x4 = apply(x2, x3)
    x5 = switch(x4, ONE, TWO)
    O = cmirror(x5)
    return O


def solve_00d62c1b(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, ZERO)
    x3 = rbind(bordering, I)
    x4 = compose(flip, x3)
    x5 = mfilter(x2, x4)
    O = fill(I, FOUR, x5)
    return O


def solve_7b6016b9(I):
    x1 = objects(I, T, F, F)
    x2 = rbind(bordering, I)
    x3 = compose(flip, x2)
    x4 = mfilter(x1, x3)
    x5 = fill(I, TWO, x4)
    O = replace(x5, ZERO, THREE)
    return O


def solve_67385a82(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, THREE)
    x3 = sizefilter(x2, ONE)
    x4 = difference(x2, x3)
    x5 = merge(x4)
    O = fill(I, EIGHT, x5)
    return O


def solve_a5313dff(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, ZERO)
    x3 = rbind(bordering, I)
    x4 = compose(flip, x3)
    x5 = mfilter(x2, x4)
    O = fill(I, ONE, x5)
    return O


def solve_ea32f347(I):
    x1 = objects(I, T, F, T)
    x2 = replace(I, FIVE, FOUR)
    x3 = argmin(x1, size)
    x4 = argmax(x1, size)
    x5 = fill(x2, ONE, x4)
    O = fill(x5, TWO, x3)
    return O


def solve_d631b094(I):
    x1 = palette(I)
    x2 = other(x1, ZERO)
    x3 = ofcolor(I, x2)
    x4 = size(x3)
    x5 = astuple(ONE, x4)
    O = canvas(x2, x5)
    return O


def solve_10fcaaa3(I):
    x1 = leastcolor(I)
    x2 = hconcat(I, I)
    x3 = vconcat(x2, x2)
    x4 = ofcolor(x3, x1)
    x5 = mapply(ineighbors, x4)
    O = underfill(x3, EIGHT, x5)
    return O


def solve_007bbfb7(I):
    x1 = hupscale(I, THREE)
    x2 = vupscale(x1, THREE)
    x3 = hconcat(I, I)
    x4 = hconcat(x3, I)
    x5 = vconcat(x4, x4)
    x6 = vconcat(x5, x4)
    O = cellwise(x2, x6, ZERO)
    return O


def solve_496994bd(I):
    x1 = width(I)
    x2 = height(I)
    x3 = halve(x2)
    x4 = astuple(x3, x1)
    x5 = crop(I, ORIGIN, x4)
    x6 = hmirror(x5)
    O = vconcat(x5, x6)
    return O


def solve_1f876c06(I):
    x1 = fgpartition(I)
    x2 = compose(last, first)
    x3 = power(last, TWO)
    x4 = fork(connect, x2, x3)
    x5 = fork(recolor, color, x4)
    x6 = mapply(x5, x1)
    O = paint(I, x6)
    return O


def solve_05f2a901(I):
    x1 = objects(I, T, F, T)
    x2 = colorfilter(x1, TWO)
    x3 = first(x2)
    x4 = colorfilter(x1, EIGHT)
    x5 = first(x4)
    x6 = gravitate(x3, x5)
    O = move(I, x3, x6)
    return O


def solve_39a8645d(I):
    x1 = objects(I, T, T, T)
    x2 = totuple(x1)
    x3 = apply(color, x2)
    x4 = mostcommon(x3)
    x5 = matcher(color, x4)
    x6 = extract(x1, x5)
    O = subgrid(x6, I)
    return O


def solve_1b2d62fb(I):
    x1 = lefthalf(I)
    x2 = righthalf(I)
    x3 = ofcolor(x1, ZERO)
    x4 = ofcolor(x2, ZERO)
    x5 = intersection(x3, x4)
    x6 = replace(x1, NINE, ZERO)
    O = fill(x6, EIGHT, x5)
    return O


def solve_90c28cc7(I):
    x1 = objects(I, F, F, T)
    x2 = first(x1)
    x3 = subgrid(x2, I)
    x4 = dedupe(x3)
    x5 = rot90(x4)
    x6 = dedupe(x5)
    O = rot270(x6)
    return O


def solve_b6afb2da(I):
    x1 = objects(I, T, F, F)
    x2 = replace(I, FIVE, TWO)
    x3 = colorfilter(x1, FIVE)
    x4 = mapply(box, x3)
    x5 = fill(x2, FOUR, x4)
    x6 = mapply(corners, x3)
    O = fill(x5, ONE, x6)
    return O


def solve_b9b7f026(I):
    x1 = objects(I, T, F, F)
    x2 = argmin(x1, size)
    x3 = rbind(adjacent, x2)
    x4 = remove(x2, x1)
    x5 = extract(x4, x3)
    x6 = color(x5)
    O = canvas(x6, UNITY)
    return O


def solve_ba97ae07(I):
    x1 = objects(I, T, F, T)
    x2 = totuple(x1)
    x3 = apply(color, x2)
    x4 = mostcommon(x3)
    x5 = ofcolor(I, x4)
    x6 = backdrop(x5)
    O = fill(I, x4, x6)
    return O


def solve_c9f8e694(I):
    x1 = height(I)
    x2 = width(I)
    x3 = ofcolor(I, ZERO)
    x4 = astuple(x1, ONE)
    x5 = crop(I, ORIGIN, x4)
    x6 = hupscale(x5, x2)
    O = fill(x6, ZERO, x3)
    return O


def solve_d23f8c26(I):
    x1 = asindices(I)
    x2 = width(I)
    x3 = halve(x2)
    x4 = matcher(last, x3)
    x5 = compose(flip, x4)
    x6 = sfilter(x1, x5)
    O = fill(I, ZERO, x6)
    return O


def solve_d5d6de2d(I):
    x1 = objects(I, T, F, T)
    x2 = sfilter(x1, square)
    x3 = difference(x1, x2)
    x4 = compose(backdrop, inbox)
    x5 = mapply(x4, x3)
    x6 = replace(I, TWO, ZERO)
    O = fill(x6, THREE, x5)
    return O


def solve_dbc1a6ce(I):
    x1 = ofcolor(I, ONE)
    x2 = product(x1, x1)
    x3 = fork(connect, first, last)
    x4 = apply(x3, x2)
    x5 = fork(either, vline, hline)
    x6 = mfilter(x4, x5)
    O = underfill(I, EIGHT, x6)
    return O


def solve_ded97339(I):
    x1 = ofcolor(I, EIGHT)
    x2 = product(x1, x1)
    x3 = fork(connect, first, last)
    x4 = apply(x3, x2)
    x5 = fork(either, vline, hline)
    x6 = mfilter(x4, x5)
    O = underfill(I, EIGHT, x6)
    return O


def solve_ea786f4a(I):
    x1 = width(I)
    x2 = shoot(ORIGIN, UNITY)
    x3 = decrement(x1)
    x4 = tojvec(x3)
    x5 = shoot(x4, DOWN_LEFT)
    x6 = combine(x2, x5)
    O = fill(I, ZERO, x6)
    return O
