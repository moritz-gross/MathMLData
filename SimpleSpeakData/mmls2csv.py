###
#  Given the MathML in some file/dir build up parent/child relationships and write a csv file with those.
###
from lxml import etree    # pip install lxml
import os
import numpy as np        # pip install numpy
import csv

# for debugging
import sys
sys.stdout.reconfigure(encoding='utf-8')


def get_mathml_from_file(html_file: str) -> list[etree._Element]:
    '''Returns all the MathML found in the file'''
    file_name_tail = os.path.split(html_file)[1]  # remember where the expr came from
    answer = []
    for _, math in etree.iterparse(html_file,
                                   tag='math',
                                   remove_blank_text=True,
                                   remove_comments=True,
                                   encoding='UTF-8',
                                   html=True
                                   ):
        math.set('data-file', file_name_tail)
        answer.append(math)
    return answer


def get_mathml_from_dir(dir: str) -> list[etree._Element]:
    '''Returns all the MathML found in .html files inside "dir" and all dirs (recursively) inside "dir".
       Also works if "dir" is a file'''
    if os.path.isfile(dir):
        return get_mathml_from_file(dir)
    mathml = []
    for entry in os.listdir(dir):
        full_path = os.path.join(dir, entry)
        if os.path.isdir(full_path):
            mathml.extend(get_mathml_from_dir(full_path))
        elif entry.endswith('.html'):
            mathml.extend(get_mathml_from_file(full_path))
    return mathml


def write_simplified_mathml(exprs: list[etree._Element], out_file: str):
    '''Cleans up the MathML by removing non-presentation MathML, mpadded, mstyle,
       and other non-spoken MathML elements.
       Also cleans mrows with one child'''
    with open(out_file, 'w', encoding='utf8') as out_stream:
        for mathml in exprs:
            # print(f"\nwrite_simplified_mathml: AFTER -- {clean_mathml(mathml)}")
            cleaned_mathml = clean_mathml(mathml)
            if cleaned_mathml is not None:
                out_stream.write(etree.tostring(cleaned_mathml, encoding="unicode"))
                out_stream.write('\n')


def clean_mathml(mathml: etree._Element) -> etree._Element | None:
    '''Cleans up the input.
       There is a weird special case from LaTeXML where math is inside of html inside of math:
       in the annotation-xml of the outer math, there is no semantics around the Content MathML,
         we detect that and return None
    '''
    if len(mathml.getchildren()) == 0:
        # empty math tag
        return None
    if mathml[0].tag == 'apply':
        return None
    mathml.tail = None
    return clean_mathml_children(mathml)


def wrap_children_with_mrow_if_needed(element: etree._Element) -> etree._Element:
    '''Change inferred mrows to explict mrows'''
    if (
        len(element.getchildren()) > 1 and
        element.tag in ['math', 'sqrt', 'mstyle', 'merror', 'mpadded', 'mphantom', 'menclose', 'mtd', 'mscarry']
    ):
        # wrap with mrow
        new_mrow = etree.Element('mrow')
        for child in element.getchildren():
            new_mrow.append(child)  # append removes it from 'element'
        element.append(new_mrow)
    return element


def clean_mathml_children(element: etree._Element) -> etree._Element:
    element.tail = None
    element_attr = element.attrib
    element_attr.pop('xmlns', None)
    element_attr.pop('alttext', None)
    element_attr.pop('class', None)
    element_attr.pop('display', None)
    element_attr.pop('id', None)
    element_attr.pop('xref', None)
    element_name = element.tag
    if element_name in LEAF_ELEMENTS:
        # could have HTML inside -- don't search that -- turn it into text
        text = element.text
        if text is None:
            text = etree.tostring(element, encoding='unicode').removesuffix(f"</{element_name}>").removeprefix(f"<{element_name}>")
        element.text = text.replace("\n", r"\n")
        for other_xml in element:
            element.remove(other_xml)     # is there a better way to delete all the chldren?
        return element

    # this makes sure that mstyle, etc., only have one child
    element = wrap_children_with_mrow_if_needed(element)
    for i, child in enumerate(element):
        child = clean_mathml_children(child)
        child_name = child.tag
        if child_name in ['semantics', 'mstyle', 'mpadded', 'mphantom']:
            # this relies on presentation MathML being the only child of semantics that isn't annotation/annotation-xml
            if len(child) == 0:
                element.remove(child)
            else:
                new_child = child[0]
                element.remove(child)
                element.insert(i, new_child)
                if child_name == 'semantics' and new_child.tag == 'annotation-xml':
                    # encoding= 'annotation-xml' because we removed the other ones -- remove it
                    grand_child = new_child[0]
                    element.remove(new_child)
                    element.insert(i, grand_child)


        elif (child_name == 'annotation' or
              (child_name == 'annotation-xml' and child.get('encoding', '') != 'MathML-Presentation')):
            element.remove(child)
    # clean up extra mrows (recursively)
    for i, child in enumerate(element):
        while child.tag == 'mrow' and len(child) == 1:
            grandchild = child[0]
            element.remove(child)
            element.insert(i, grandchild)
            child = grandchild
    return element



# =========== Compute Stats ==============
# The Counts dictionary represents the parent/child data
# A DictEntry represents the child relationship
# E.g., a mfrac with an mrow numerator and a mn denominator which has happened 3 times would look like
#   {'mfrac': {['mrow/3; mn']: 3}}
# For leaves there is only one entry (the contents).
# Here's an example:
#   {'mi': {'a': 12, 'x': 23}}
type DictEntry = dict[str, int]
type Counts = dict[str, DictEntry]

SIMPLE_SPEAK_TAGS = ['<mfrac', '<msup', '<msub', '<msubsup', '<msqrt', '<mroot', '<munder']


def compute_mathml_stats(in_dir: str, counts: Counts = {}) -> int:
    '''Returns all the MathML found in .mmls files inside "dir" and all dirs (recursively) inside "dir".
       Also works if "dir" is a file'''
    n_tags = 0
    if os.path.isfile(in_dir):
        n_tags = compute_mathml_stats_from_file(in_dir, counts)
    else:
        for entry in os.listdir(in_dir):
            full_path = os.path.join(in_dir, entry)
            if os.path.isdir(full_path):
                print(f'compute_mathml_stats_from_dir: {full_path}...')
                n_tags += compute_mathml_stats(full_path, counts)
            elif entry.endswith('.mmls'):
                n_tags += compute_mathml_stats_from_file(full_path, counts)
    return n_tags

def compute_mathml_stats_from_file(file: str, counts: Counts) -> int:
    '''Reads a file that consists of MathML expressions, one per line ("xxx.mmls file")'''
    with open(file, 'r', encoding='utf8') as in_stream:
        n_tags = 0
        lines = in_stream.readlines()
        for line in lines:
            mathml = etree.fromstring(line)
            n_tags += get_stats_from_mathml(mathml, counts)
    return n_tags

LEAF_ELEMENTS = ['mi', 'mo', 'mn', 'mtext', 'none', 'mprescripts', 'mspace']


def get_stats_from_mathml(element: etree._Element, counts: Counts) -> int:
    '''Recursively set "counts"'''
    # Doing counts for all the children of mrows is about 75% of the counts (including leaves) and isn't that useful
    # If the mrow has more than 3 children, put the counts into a "#children>3" bin
    n_tags = 1
    name = element.tag
    name_entry = counts.get(name, {})
    if name in LEAF_ELEMENTS:
        contents = element.text
        name_entry[contents] = name_entry.get(contents, 0) + 1
    else:
        children_key = ''
        for child in element:
            if children_key != '':
                children_key += '; '
            child_name = child.tag
            if child_name in LEAF_ELEMENTS:
                children_key += child_name
            else:
                children_key += f"{child_name}/{len(child)}"
            n_tags += get_stats_from_mathml(child, counts)
        if name == 'mrow' and len(element.getchildren()) > 3:
            children_key = "children>3"
        name_entry[children_key] = name_entry.get(children_key, 0) + 1
    counts[name] = name_entry
    return n_tags


def compute_mathml_node_stats(in_dir: str, internal: dict[int, int],
                              all: dict[int, int],
                              weighted: dict[int, int],
                              simple_weighted: dict[int, int],
                              savings: list[tuple[int, int]]) -> int:
    '''Returns all the MathML found in .html files inside "dir" and all dirs (recursively) inside "dir".
       Also works if "dir" is a file'''
    if os.path.isfile(in_dir):
        return compute_mathml_node_stats_from_file(in_dir, internal, all, weighted, simple_weighted, savings)
    else:
        n_exprs = 0
        for entry in os.listdir(in_dir):
            full_path = os.path.join(in_dir, entry)
            if os.path.isdir(full_path):
                n_exprs += compute_mathml_node_stats_from_file(full_path,
                                                               internal, all, weighted, simple_weighted,
                                                               savings)
            elif entry.endswith('.mmls'):
                n_exprs += compute_mathml_node_stats_from_file(full_path,
                                                               internal, all, weighted, simple_weighted,
                                                               savings)
        return n_exprs


def compute_mathml_node_stats_from_file(file: str,
                                        internal: dict[int, int],
                                        all: dict[int, int],
                                        weighted: dict[int, int],
                                        simple_weighted: dict[int, int],
                                        savings: list[tuple[int, int]]) -> int:
    '''Reads a file that consists of MathML expressions, one per line.
       Returns the number of included and excluded lines'''
    # print(f'compute_mathml_node_stats_from_file: {file}...')
    with open(file, 'r', encoding='utf8') as in_stream:
        lines = in_stream.readlines()
        n_exprs = 0
        for line in lines:
            mathml = etree.fromstring(line)
            i, a, w, sw = get_node_stats_from_mathml(mathml)
            if i == 1:      # TEMPORARY to compute values that don't include trivial exprs
                continue
            if any(tag in line for tag in SIMPLE_SPEAK_TAGS):   # little hack to only consider SimpleSpeak tags
                savings.append((sw, w))
            internal[i] = internal.get(i, 0) + 1
            all[a] = all.get(a, 0) + 1
            weighted[w] = weighted.get(w, 0) + 1
            simple_weighted[sw] = simple_weighted.get(sw, 0) + 1
            n_exprs += 1
        return n_exprs


NODE_WEIGHTS_SIMPLE: dict[str, int] = {
    'mfrac': 2,
    'msub': 1,
    'msup': 1,
    'msubsup': 2,
    'msqrt': 3,
    'mroot': 2,
    'mtr': 1,
    'mlabeledtr': 4,
    'mtd': 2,
    'mtable': 8,
    'munder': 3,
    'munderover': 6,
    'menclose': 4,
    'mover': 0,  # e.g, "x bar" -- mover adds nothing
    'mmultiscripts': 5,
}


NODE_WEIGHTS_NOT_SIMPLE: dict[str, int] = {
    'mfrac': 7,
    'msub': 3,
    'msup': 4,
    'msubsup': 5,
    'msqrt': 5,
    'mroot': 4,
    'mtr': 1,
    'mlabeledtr': 4,
    'mtd': 2,
    'mtable': 8,
    'munder': 6,
    'munderover': 9,
    'menclose': 4,
    'mover': 0,  # e.g, "x bar" -- mover adds nothing
    'mmultiscripts': 5,
}


def get_node_stats_from_mathml(element: etree._Element) -> tuple[int, int, int, int]:
    '''Recursively compute the number nodes: internal only, all, weighted all, simple_weighted'''
    name = element.tag
    if name == 'mi':
        return (0, 1, 1, 1)
    elif name == 'mn':
        return (0, 1, 2*len(element.text)-1, 2*len(element.text)-1)
    elif name == 'mo':
        return (0, 1, 2, 2)
    elif name == 'mtext':
        return (0, 1, len(element.text.split(' ')), len(element.text.split(' ')))
    elif name == 'mspace':
        return (0, 0, 0, 0)
    else:
        internal = all = 1
        weighted = NODE_WEIGHTS_NOT_SIMPLE.get(name, 0)
        simple_weighted = NODE_WEIGHTS_SIMPLE.get(name, 0)
        for child in element.getchildren():
            i, a, w, sw = get_node_stats_from_mathml(child)
            internal += i
            all += a
            weighted += w
            simple_weighted += sw
        return (internal, all, weighted, simple_weighted)


def write_stats(out_file: str, counts: Counts, n_tags: int):
    with open(out_file, 'w', encoding='utf8') as out_stream:
        out_stream.write(f"name[child1/grandchild_count;...],frequency({n_tags})\n")
        for name, children in counts.items():
            # print in sorted by count order
            for child in sorted(children, key=children.get, reverse=True):
                out_stream.write(f"{name}[{child}],{children[child]}\n")


def write_node_stats(out_file: str,
                     n_exprs: int,
                     n_savings: int,
                     internal: dict[int, int],
                     all: dict[int, int],
                     weighted: dict[int, int],
                     simple_weighted: dict[int, int],
                     full_savings: dict[int, int],
                     simple_savings: dict[int, int]):
    with open(out_file, 'w', encoding='utf8') as out_stream:
        write_node_stats_for_count(out_stream, "Weighted", weighted, n_exprs)
        write_node_stats_for_count(out_stream, "SimpleWeighted", simple_weighted, n_exprs)
        write_node_stats_for_count(out_stream, "Full Weighted", full_savings, n_savings)
        write_node_stats_for_count(out_stream, "Simple Savings", simple_savings, n_savings)
        # write_node_stats_for_count(out_stream, "Internal", internal, n_exprs)
        # write_node_stats_for_count(out_stream, "All", all, n_exprs)


def write_node_stats_for_count(out, header: str, data: dict[int, int], n_exprs: int):
    out.write(f'{header}, #Exprs={n_exprs}\n')
    sorted_data = sorted(data.items())
    cummulative = 0
    for key, value in sorted_data:
        if key == 0:        # there are a few empty nodes -- skip them as errors
            continue
        cummulative += value
        out.write(f"{key}, {value}, {round(100*cummulative/n_exprs, )}\n")
    out.write('\n\n')


def write_file_for_simplfied_mathml(in_dir: str, out_file: str):
    write_simplified_mathml(get_mathml_from_dir(in_dir), out_file)


def write_dir_for_simplfied_mathml():
    in_root = r"D:\Dev\SimpleSpeakData\arXiv"
    out_root = r"C:\Dev\SimpleSpeakData\arXiv"
    for entry in os.listdir(in_root):
        print(f"Working on {entry}...")
        in_dir = os.path.join(in_root, entry)
        if os.path.isdir(in_dir):
            out_file_name = entry + '.mmls'
            write_file_for_simplfied_mathml(in_dir, os.path.join(out_root, out_file_name))
        elif entry.endswith('.mmls'):
            print(f"WARNING: found none directory '{in_dir}")


def get_internal_all_weighted_totals(file_name: str) -> tuple[int, int, int]:
    '''Reads the distribution .csv and looks for the totals (which are headings).
       This speeds things up a lot by avoiding reading all the .mmls files
    '''
    with open(file_name, 'r', encoding='utf8') as in_stream:
        lines = in_stream.readlines()
        internal = all = weighted = 0
        for line in lines:
            if line.startswith('Internal, #Nodes='):
                internal = int(line.split('=')[1])
            elif line.startswith('All, #Nodes='):
                all = int(line.split('=')[1])
            elif line.startswith('Weighted, #Nodes='):
                weighted = int(line.split('=')[1])
                return (internal, all, weighted)
        raise Exception(f"get_internal_all_weighted_totals: didn't find total in {file_name}")


def stats(in_file: str, data: str, distribution: str, savings_file: str):
    # counts = {}
    # n_tags = compute_mathml_stats(in_file, counts)
    # write_stats(data, counts, n_tags)

    internal: dict[int, int] = {}
    all: dict[int, int] = {}
    weighted: dict[int, int] = {}
    simple_weighted: dict[int, int] = {}
    savings: list[tuple[int, int]] = []
    print(f'compute_mathml_node_stats for {in_file}...')
    n_exprs = compute_mathml_node_stats(in_file, internal, all, weighted, simple_weighted, savings)
    simple_savings: dict[int, int] = {}
    full_savings: dict[int, int] = {}
    for (simple, full) in savings:
        simple_savings[simple] = simple_savings.get(simple, 0) + 1
        full_savings[full] = full_savings.get(full, 0) + 1

    # write_savings(savings_file, savings, n_exprs)
    write_node_stats(distribution, n_exprs, len(savings), internal, all,
                     weighted, simple_weighted,
                     full_savings, simple_savings)


def write_savings(out_file: str, savings: list[tuple[int, int]], n_exprs: int):
    with open(out_file, 'w', encoding='utf8') as out_stream:
        n_included_exprs = len(savings)
        ratio = n_included_exprs/(n_exprs)
        out_stream.write(f'#included/excluded exprs = {n_included_exprs}/{n_exprs} ({100*round(ratio, 3)}%)\n')
        fractional_savings = list(x[0]/x[1] for x in savings)
        fractional_mean = round(100*np.mean(fractional_savings), 0)
        fractional_median = round(100*np.median(fractional_savings), 0)
        out_stream.write(f'fractional syllable savings mean={fractional_mean}%; median = {fractional_median}%\n')
        word_savings = list(x[1] - x[0] for x in savings)
        out_stream.write(f'syllable savings mean={round(np.mean(word_savings), 0)}; median={np.median(word_savings)}\n')
        non_simple_weights = list(x[1] for x in savings)
        out_stream.write(f'non_simple mean={round(np.mean(non_simple_weights), 0)}; median={np.median(non_simple_weights)}\n')



def test_clean_math():
    clean_mathml_test = [
        # (input, expected-output)
        ("<math><mi>a</mi></math>", "<math><mi>a</mi></math>"),
        ("<math><mi>a</mi><mi>b</mi></math>", "<math><mrow><mi>a</mi><mi>b</mi></mrow></math>"),
        ("<math><mstyle><mi>a</mi></mstyle></math>", "<math><mi>a</mi></math>"),
        ("<math><mstyle><mrow><mi>a</mi></mrow><mi>b</mi></mstyle></math>",
         "<math><mrow><mi>a</mi><mi>b</mi></mrow></math>"),
        ("<math><mrow><mstyle><mrow><mi>a</mi></mrow><mi>b</mi></mstyle></mrow></math>",
         "<math><mrow><mi>a</mi><mi>b</mi></mrow></math>"),
        ("<math><mrow><mstyle><mi>a</mi><mi>b</mi></mstyle></mrow></math>",
         "<math><mrow><mi>a</mi><mi>b</mi></mrow></math>"),
        ("<math><mstyle><mi>a</mi></mstyle><mstyle><mi>b</mi></mstyle></math>",
         "<math><mrow><mi>a</mi><mi>b</mi></mrow></math>"),
        ("<math><mstyle><mi>a</mi><mi>a2</mi></mstyle><mstyle><mi>b</mi><mi>b2</mi></mstyle></math>",
         "<math><mrow><mrow><mi>a</mi><mi>a2</mi></mrow><mrow><mi>b</mi><mi>b2</mi></mrow></mrow></math>"),
        ("<math><mrow><mstyle><mi>a</mi><mi>a2</mi></mstyle><mstyle><mi>b</mi><mi>b2</mi></mstyle></mrow></math>",
         "<math><mrow><mrow><mi>a</mi><mi>a2</mi></mrow><mrow><mi>b</mi><mi>b2</mi></mrow></mrow></math>"),
        ("<math><semantics><mstyle><mi>a</mi></mstyle></semantics></math>",
         "<math><mi>a</mi></math>"),
        ("""<math><semantics><mstyle><mi>a</mi></mstyle>\
                        <annotation-xml encoding='content'><ci>a</ci></annotation-xml></semantics></math>""",
         "<math><mi>a</mi></math>"),
        ("""<math><semantics><annotation-xml encoding='MathML-Presentation'><mstyle><mi>a</mi></mstyle></annotation-xml>
                <annotation-xml encoding='content'><ci>a</ci></annotation-xml></semantics></math>""",
         "<math><mi>a</mi></math>"),
        ("<math><semantics><mi>a</mi></semantics></math>",
         "<math><mi>a</mi></math>"),
        ]
    n_errors = 0
    for input, output in clean_mathml_test:
        try:
            cleaned_mathml = etree.tostring(clean_mathml(etree.fromstring(input)), encoding="unicode")
            if cleaned_mathml != output:
                n_errors += 1
                print(f"\nError in clean test\n   Input: {input}")
                print(f"Expected: {output}")
                print(f" Cleaned: {cleaned_mathml}")
        except Exception as e:
            print(f"clean_mathml() error on input:\n{input}")
            print(f"Error is: {e}")
    print(f"test_clean_math: {n_errors} errors in {len(clean_mathml_test)} tests.")


def create_excel_plot_data(in_file: str, out_file: str):
    with open(in_file, 'r', encoding='utf8') as in_stream:
        csv_reader = csv.reader(in_stream, delimiter=',')
        num_entries = 20
        hs_full = [0]*num_entries
        hs_simple = [0]*num_entries
        college_full = [0]*num_entries
        college_simple = [0]*num_entries
        arxiv_full = [0]*num_entries
        arxiv_simple = [0]*num_entries
        for line in csv_reader:
            i = int(line[0]) - 1
            hs_full[i] = int(line[1])
            hs_simple[i] = int(line[2])
            college_full[i] = int(line[3])
            college_simple[i] = int(line[4])
            arxiv_full[i] = int(line[5])
            arxiv_simple[i] = int(line[6])

    min_hs = min( hs_simple[i] - hs_full[i] for i in range(5, 20))
    min_col = min( college_simple[i] - college_full[i] for i in range(5, 20))
    min_art = min( arxiv_simple[i] - arxiv_full[i] for i in range(5, 20))
    max_hs = max( hs_simple[i] - hs_full[i] for i in range(5, 20))
    max_col = max( college_simple[i] - college_full[i] for i in range(5, 20))
    max_art = max( arxiv_simple[i] - arxiv_full[i] for i in range(5, 20))

    print(f'hs: min/max = {min_hs}/{max_hs}, col: min/max = {min_col}/{max_col}, arxiv: min/max = {min_art}/{max_art}')
    # make HS, Col, and article be the major group with counts repeating
    with open(out_file, 'w', encoding='utf8') as out_stream:
        out_stream.write(', , Full, Simple (extra), Total\n')
        out_stream.write('\n')
        for i in range(5, 11):
            label = '        High School' if i == 7 else ''      # extra spaces to help center it  
            out_stream.write(f'{label}, {i+1}, {hs_full[i]}, {hs_simple[i] - hs_full[i]}, {hs_simple[i]}\n')
        out_stream.write('\n')
        for i in range(5, 11):
            label = '        College' if i == 7 else ''
            out_stream.write(f'{label}, {i+1}, {college_full[i]}, {college_simple[i] - college_full[i]}, {college_simple[i]}\n')
        out_stream.write('\n')
        for i in range(5, 11):
            label = '        Article' if i == 7 else ''
            out_stream.write(f'{label}, {i+1}, {arxiv_full[i]}, {arxiv_simple[i] - arxiv_full[i]}, {arxiv_simple[i]}\n')

    # groups HS, Col, and Article together under each syllable count
    # with open(out_file, 'w', encoding='utf8') as out_stream:
    #     out_stream.write(', , Full, Simple (extra)\n')
    #     for i in range(num_entries):
    #         out_stream.write(f' , HS, {hs_full[i]}, {hs_simple[i] - hs_full[i]}, {hs_simple[i]}\n')
    #         out_stream.write(f'{i+1}, Col, {college_full[i]}, {college_simple[i] - college_full[i]}, {college_simple[i]}\n')
    #         out_stream.write(f' , Art, {arxiv_full[i]}, {arxiv_simple[i] - arxiv_full[i]}, {arxiv_simple[i]}\n')
    #         out_stream.write('\n')


# test_clean_math()

# def compute_number_arXiv_articles():
#     dir = r"D:\Dev\SimpleSpeakData\arXiv-file-counts\arXiv"
#     print(f"# files = {len(os.listdir(dir))}")
#     count = 0
#     for entry in os.listdir(dir):
#         file = dir + '/' + entry
#         with open(file, 'r', encoding='utf8') as in_stream:
#             lines = in_stream.readlines()
#             count += int(lines[0].strip())

#     print(f"Count is {count}")

# compute_number_arXiv_articles()

# for testing
# write_file_for_simplfied_mathml(
#     # r"C:\Dev\SimpleSpeakData\test.html",
#     r"D:\Dev\SimpleSpeakData\arXiv\0008",
#     r"C:\Dev\SimpleSpeakData\test.out"
# )

# print(
#     len(get_mathml_from_dir(r'C:\Users\neils\Dropbox\ar5iv-0001\astro-ph0001369.html'))
# )


# stats(
#     r"C:\Dev\SimpleSpeakData\ebooks",
#     r"C:\Dev\SimpleSpeakData\ebooks-data.csv",
#     r"C:\Dev\SimpleSpeakData\ebooks-distribution.csv",
#     r"C:\Dev\SimpleSpeakData\ebooks-savings.txt"
# )
# stats(
#     r"C:\Dev\SimpleSpeakData\open-stax",
#     r"C:\Dev\SimpleSpeakData\open-stax-data.csv",
#     r"C:\Dev\SimpleSpeakData\open-stax-distribution.csv",
#     r"C:\Dev\SimpleSpeakData\open-stax-savings.txt"
# )
# stats(
#     r"C:\Dev\SimpleSpeakData\test.mmls",
#     r"C:\Dev\SimpleSpeakData\test-data.csv",
#     r"C:\Dev\SimpleSpeakData\test-distribution.csv"
#     r"C:\Dev\SimpleSpeakData\test-savings.csv"
# )
# stats(
#     r"C:\Dev\SimpleSpeakData\highschool",
#     r"C:\Dev\SimpleSpeakData\highschool-data.csv",
#     # r"C:\Dev\SimpleSpeakData\highschool-distribution.csv",
#     r"C:\Dev\SimpleSpeakData\highschool-distribution-non-trivial.csv",
#     r"C:\Dev\SimpleSpeakData\highschool-savings.txt"
# )
# stats(
#     r"C:\Dev\SimpleSpeakData\college",
#     r"C:\Dev\SimpleSpeakData\college-data.csv",
#     # r"C:\Dev\SimpleSpeakData\college-distribution.csv",
#     r"C:\Dev\SimpleSpeakData\college-distribution-non-trivial.csv",
#     r"C:\Dev\SimpleSpeakData\college-savings.txt"
# )
# stats(
#     r"C:\Dev\SimpleSpeakData\arXiv",
#     r"C:\Dev\SimpleSpeakData\arXiv-data.csv",
#     # r"C:\Dev\SimpleSpeakData\arXiv-distribution.csv",
#     r"C:\Dev\SimpleSpeakData\arXiv-distribution-non-trivial.csv",
#     r"C:\Dev\SimpleSpeakData\arXiv-savings.txt"
# )

# create_excel_plot_data("cummulative-all.csv", "plot2-all.csv")
# create_excel_plot_data("cummulative-notations.csv", "plot2-notations.csv")
create_excel_plot_data("cummulative-nontrivial.csv", "plot2-nontrivial.csv")
# write_dir_for_simplfied_mathml()
