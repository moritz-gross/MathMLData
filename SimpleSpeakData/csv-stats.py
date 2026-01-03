'''Takes a CSV file representing arXiv stats for parent/child counts and computes the number of "simple" cases'''
import csv
import sys
import mmls2csv
sys.stdout.reconfigure(encoding='utf-8')


SIMPLE_CHILDREN = ['mi', 'mn', 'mo', 'mtext']


def get_stats_from_csv_file(file_name: str) -> tuple[dict[str, int], dict[str, int]]:
    with open(file_name, mode='r', encoding='utf8') as in_stream:
        simple: dict[str, int] = {}
        not_simple: dict[str, int] = {}
        trivial_math = 0
        csv_reader = csv.reader(in_stream, delimiter=',')
        is_header = True
        n_lines = 0
        non_trivial_math = 0
        for row in csv_reader:
            if is_header:
                # skip first line
                is_header = False
                continue

            n_lines += 1
            parent, is_simple = parse_mathml_info(row[0])
            count = int(row[-1])
            # trivial math?
            # break apart the mroot cases
            if parent == 'mroot':
                is_base_simple, is_index_simple = parse_mroot_info(row[0])
                if is_base_simple and is_index_simple:
                    simple['mroot_s_s'] = simple.get('mroot_s_s', 0) + count
                elif is_base_simple:
                    simple['mroot_s_c'] = simple.get('mroot_s_c', 0) + count
                elif is_index_simple:
                    simple['mroot_c_s'] = simple.get('mroot_c_s', 0) + count
            if is_simple:
                simple[parent] = simple.get(parent, 0) + count
            else:
                not_simple[parent] = not_simple.get(parent, 0) + count

    # some special cases -- these are subcases of mroot
    not_simple['mroot_s_s'] = not_simple['mroot_s_c'] = not_simple['mroot_c_s'] = not_simple.get('mroot', 0)
    return (simple, not_simple)


def get_trivial_math_counts_from_file(file_name: str) -> tuple[int, int]:
    with open(file_name, mode='r', encoding='utf8') as in_stream:
        csv_reader = csv.reader(in_stream, delimiter=',')
        trivial_math = non_trivial_math = 0
        is_header = True
        for row in csv_reader:
            if is_header:
                # skip first line
                is_header = False
                continue

            parent, is_simple = parse_mathml_info(row[0])
            count = int(row[-1])
            # trivial math?
            if parent == 'math':
                if any(c in row[0] for c in ['math[mi]', 'math[mn]', 'math[mo]', 'math[mtext]']):
                    trivial_math += count
                else:
                    non_trivial_math += count
    return (trivial_math, non_trivial_math)


def get_integer_subscript_counts_from_file(file_name: str) -> tuple[int, int]:
    with open(file_name, mode='r', encoding='utf8') as in_stream:
        csv_reader = csv.reader(in_stream, delimiter=',')
        trivial_math = non_trivial_math = 0
        is_header = True
        for row in csv_reader:
            if is_header:
                # skip first line
                is_header = False
                continue

            # The format of 'parent_child' is 'msubsup[mi; mi; mn]'
            parts: list[str] = row[0].strip().split('[')
            parent = parts[0]
            # trivial math?
            if parent == 'msub' or parent == 'msubsup':
                parts: list[str] = parts[-1].split(';')
                parts[-1] = parts[-1].split(']')[0].strip()   # peel off ']'
                count = int(row[-1])
                if len(parts) < 2:  # seems to be something bad in the arXiv data
                    continue
                if parts[1].strip() == 'mn':
                    trivial_math += count
                else:
                    non_trivial_math += count

    return (trivial_math, non_trivial_math)


def print_trival_math_counts():
    n_trivial, n_non_trivial = get_trivial_math_counts_from_file('highschool-data.csv')
    print(f'High school: trival={n_trivial}, total={n_trivial+n_non_trivial}, {round(100*n_trivial/(n_trivial+n_non_trivial), 1)}%')
    n_trivial, n_non_trivial = get_trivial_math_counts_from_file('college-data.csv')
    print(f'College: trival={n_trivial}, total={n_trivial+n_non_trivial}, {round(100*n_trivial/(n_trivial+n_non_trivial), 1)}%')
    n_trivial, n_non_trivial = get_trivial_math_counts_from_file('arxiv-data.csv')
    print(f'arXiv: trival={n_trivial}, total={n_trivial+n_non_trivial}, {round(100*n_trivial/(n_trivial+n_non_trivial), 1)}%')


def print_number_subscript_counts():
    n_trivial, n_non_trivial = get_integer_subscript_counts_from_file('highschool-data.csv')
    print(f'High school: number={n_trivial}, total={n_trivial+n_non_trivial}, {round(100*n_trivial/(n_trivial+n_non_trivial), 1)}%')
    n_trivial, n_non_trivial = get_integer_subscript_counts_from_file('college-data.csv')
    print(f'College: number={n_trivial}, total={n_trivial+n_non_trivial}, {round(100*n_trivial/(n_trivial+n_non_trivial), 1)}%')
    n_trivial, n_non_trivial = get_integer_subscript_counts_from_file('arxiv-data.csv')
    print(f'arXiv: number={n_trivial}, total={n_trivial+n_non_trivial}, {round(100*n_trivial/(n_trivial+n_non_trivial), 1)}%')


def parse_mathml_info(parent_child: str) -> tuple[str, bool]:
    '''Returns the parent element name and a bool saying whether all the children are simple;
       for scripts if all the scripts are simple.'''
    script_elements = ['msub', 'msup', 'msubsup']
    # The format of 'parent_child' is 'msubsup[mi; mi; mn]'
    parts: list[str] = parent_child.strip().split('[')
    parent = parts[0]
    parts: list[str] = parts[-1].split(';')
    parts[-1] = parts[-1].split(']')[0].strip()   # peel off ']'
    children_are_simple = all(child.strip() in SIMPLE_CHILDREN for child in (parts[1:] if parent in script_elements else parts))
    return (parent, children_are_simple)


def parse_mroot_info(mroot_child: str) -> tuple[bool, bool]:
    '''Returns with the base, index of the mroot are simple;
       input looks like "mroot[mrow/2; mn]"'''
    parts: list[str] = mroot_child.strip().split('[')
    parts: list[str] = parts[-1].split(';')
    parts[-1] = parts[-1].split(']')[0].strip()   # peel off ']'
    return (parts[0] in SIMPLE_CHILDREN, parts[-1] in SIMPLE_CHILDREN)


def write_stats_from_csv_file(category: str) -> int:
    '''Read category-data.csv and category-distribution.csv and write category-stats.csv'''
    with open(category+'-stats.csv', mode='w', encoding='utf8') as out_stream:
        internal_total, all_total, weighted_total = mmls2csv.get_internal_all_weighted_totals(category+'-distribution.csv')
        node_weights = mmls2csv.get_node_weights()
        simple, not_simple = get_stats_from_csv_file(category+'-data.csv')
        # header
        out_stream.write(f'{"tag,".ljust(14)} {"simple".rjust(8)}, {"simple%".rjust(7)},{"weighted".rjust(7)}, {"weight%".rjust(7)}\n')
        out_stream.write(f'{"totals,".ljust(14)} {str(internal_total).rjust(8)}, {str(all_total).rjust(7)}, {str(weighted_total).rjust(7)},\n')
        key_total = 0
        for (key, not_simple_value) in sorted(not_simple.items()):
            simple_value = simple.get(key, 0)
            if key in ['mi', 'mn', 'mo', 'mtext']:
                simple_value = not_simple_value
            key_total = simple_value + not_simple_value
            simple_frequency = simple_value/key_total
            weighted_count = simple_value * node_weights.get(key, 0)
            out_stream.write((f"{(key + ',').ljust(14)} "
                              f"{str(simple_value).rjust(8)}, "
                              f"{str(round(100*simple_frequency, 3)).rjust(7)}%,"
                              f"{str(weighted_count).rjust(7)}, "
                              f"{str(round(100*weighted_count/weighted_total, 3)).rjust(6)}%\n"
                              ))
    return key_total


def get_leaf_stats_from_csv_file(file_name: str):
    with open(file_name, mode='r', encoding='utf8') as in_stream:
        csv_reader = csv.reader(in_stream, delimiter=',')
        is_header = True
        total_count = 0
        total_non_ascii_count = 0
        counts = []
        tail_count = 0
        for row in csv_reader:
            if is_header:
                is_header = False
                continue

            info = row[0].split('\t')  # e.g., "mi	A"
            if len(info) != 2:
                continue
            char = info[-1].strip()
            if char == '​' or len(char) != 1:      # filter out zero-width-space also since that is a LaTeXML artifact
                continue        # mn, mtext, multichar mi
            count = int(row[-1])
            total_count += count
            if count <= 5:
                tail_count += count
            if not char.isascii():
                total_non_ascii_count += count
            counts.append((char, count))

        print(f'Total: all={total_count}, non-ascii={total_non_ascii_count}, tail={tail_count}')
        running_total = 0
        running_non_ascii_total = 0
        threshhold = 0.99 * total_count
        non_ascii_threshold = 0.99 * total_non_ascii_count
        for (char, count) in counts:
            mark = "  " if char.isascii() else "* "
            if not char.isascii():
                running_non_ascii_total += count
                if running_non_ascii_total > non_ascii_threshold:
                    print('--------- {round(running_non_ascii_total/total_non_ascii_count, 5)}')
            running_total += count
            if running_total > threshhold:
                print('========= {round(threshhold/running_total, 5)}')

            print(f'{mark}"{char}": {count}, {round(count/total_count, 2)}/{round(count/total_non_ascii_count, 2)}')


# write_stats_from_csv_file("arXiv")
# get_leaf_stats_from_csv_file("leaf_demo.csv")
# print_trival_math_counts()
print_number_subscript_counts()