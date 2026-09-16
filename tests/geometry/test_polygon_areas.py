import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon, box

from geohalo.geometry import cell_areas, midpoint_edges, polygon_areas


def test_latlon_box_matches_cell_areas() -> None:
    lats = np.array([-0.5, 0.5])
    lons = np.array([0.0, 1.0])
    areas = cell_areas(lats, lons, spherical=True)
    lat_edges = midpoint_edges(lats)
    lon_edges = midpoint_edges(lons)
    geom = box(float(lon_edges[0]), float(lat_edges[0]), float(lon_edges[1]), float(lat_edges[1]))
    poly = polygon_areas([geom], spherical=True)
    np.testing.assert_allclose(poly[0], areas[0, 0], rtol=1e-12)


def test_high_latitude_box_smaller_than_equator() -> None:
    equator = box(0.0, -1.0, 2.0, 1.0)
    polar = box(0.0, 70.0, 2.0, 72.0)
    eq_a, pol_a = polygon_areas([equator, polar], spherical=True)
    assert eq_a > pol_a


def test_hole_is_subtracted() -> None:
    outer = box(0.0, 0.0, 10.0, 10.0)
    hole = box(2.0, 2.0, 4.0, 4.0)
    with_hole = Polygon(outer.exterior.coords, [list(hole.exterior.coords)])
    outer_a, hole_a, net_a = polygon_areas([outer, hole, with_hole], spherical=True)
    np.testing.assert_allclose(net_a, outer_a - hole_a, rtol=1e-12)


def test_multipolygon_sums_parts() -> None:
    a = box(0.0, 0.0, 1.0, 1.0)
    b = box(10.0, 10.0, 11.0, 11.0)
    multi = MultiPolygon([a, b])
    aa, ba, ma = polygon_areas([a, b, multi], spherical=True)
    np.testing.assert_allclose(ma, aa + ba, rtol=1e-12)


def test_empty_and_non_polygon_are_zero() -> None:
    out = polygon_areas([Polygon(), Point(0, 0)], spherical=True)
    np.testing.assert_array_equal(out, [0.0, 0.0])


def test_planar_uses_shapely_area() -> None:
    geom = box(0.0, 0.0, 2.0, 3.0)
    out = polygon_areas([geom], spherical=False)
    np.testing.assert_allclose(out[0], geom.area)
