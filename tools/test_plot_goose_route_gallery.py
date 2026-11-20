#!/usr/bin/env python3
"""Selection-contract tests for the GOOSE route gallery."""

from plot_goose_route_gallery import select_route_midpoints


def main():
    order = (('train', 'route_a'), ('val', 'route_b'))
    rows = []
    for split, route in order:
        for index in range(5):
            rows.append({
                '_split': split,
                'sequence': route,
                'frame_index': index,
                'token': '{}_{}'.format(route, index),
            })
    selected = select_route_midpoints(rows, order)
    assert [row['frame_index'] for row in selected] == [2, 2]
    assert len({(row['_split'], row['sequence']) for row in selected}) == 2
    try:
        select_route_midpoints(rows, (('train', 'missing'),))
    except ValueError as error:
        assert 'Missing route' in str(error)
    else:
        raise AssertionError('Missing route did not fail')
    print('GOOSE route gallery selection tests passed')


if __name__ == '__main__':
    main()
