"""
8/31/18 This is still a mess and I will deal with it at some point
Right now, I don't want to deal with the flashbacks from when I had to write this code.
"""

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from matplotlib.patches import Rectangle
import numpy as np
import brewer2mpl
from pandas import DataFrame
import matplotlib.style as style
from scipy.interpolate import spline
import json
import gc

style.use('fivethirtyeight')


from tobacco.results_storage.results_storage_get_result import get_frequencies_results

# Colorblind-friendly colors

COLORSETS = {
    'colorblind':[[0,0,0], [230/255,159/255,0], [86/255,180/255,233/255], [0,158/255,115/255],
          [213/255,94/255,0], [0,114/255,178/255]],
    '538':['#008fd5', '#fc4f30', '#e5ae38', '#6d904f', '#8b8b8b', '#810f7c'],
    'brewer_qual_1': brewer2mpl.get_map('Set1', 'qualitative', 9).mpl_colors + brewer2mpl.get_map('Dark2', 'qualitative', 8).mpl_colors,    # good
    'brewer_qual_2': brewer2mpl.get_map('Set2', 'qualitative', 8).mpl_colors + brewer2mpl.get_map('Dark2', 'qualitative', 8).mpl_colors,    # fine
    'brewer_qual_3': brewer2mpl.get_map('Accent', 'qualitative', 8).mpl_colors + brewer2mpl.get_map('Dark2', 'qualitative', 8).mpl_colors,  # ok
    'brewer_qual_4': brewer2mpl.get_map('Set3', 'qualitative', 12).mpl_colors + brewer2mpl.get_map('Dark2', 'qualitative', 8).mpl_colors,   # ok
#    'brewer_qual_5': brewer2mpl.get_map('Paired', 'qualitative', 12).mpl_colors + brewer2mpl.get_map('Dark2', 'qualitative', 8).mpl_colors,
#    'brewer_qual_6': brewer2mpl.get_map('Pastel1', 'qualitative', 9).mpl_colors + brewer2mpl.get_map('Dark2', 'qualitative', 8).mpl_colors, # ok
#    'brewer_qual_7': brewer2mpl.get_map('Pastel2', 'qualitative', 8).mpl_colors + brewer2mpl.get_map('Dark2', 'qualitative', 8).mpl_colors,
#    'brewer_qual_8': brewer2mpl.get_map('Pairedo', 'qualitative', 12).mpl_colors + brewer2mpl.get_map('Dark2', 'qualitative', 8).mpl_colors,
    'brewer_qual_9': brewer2mpl.get_map('Dark2', 'qualitative', 8).mpl_colors + brewer2mpl.get_map('Dark2', 'qualitative', 8).mpl_colors,

    'brewer_grayscale': brewer2mpl.get_map('Greys', 'Sequential', 9).mpl_colors[::-1] + brewer2mpl.get_map('Greys', 'sequential', 9).mpl_colors[::-1],
    'brewer_div_1': brewer2mpl.get_map('Spectral', 'Diverging', 11).mpl_colors[::-1] + brewer2mpl.get_map('Greys', 'sequential', 9).mpl_colors[::-1],
    'brewer_div_2': brewer2mpl.get_map('BrBg', 'Diverging', 11).mpl_colors[::-1] + brewer2mpl.get_map('Greys', 'sequential', 9).mpl_colors[::-1],
    'brewer_div_3': brewer2mpl.get_map('PRGn', 'Diverging', 11).mpl_colors[::-1] + brewer2mpl.get_map('Greys', 'sequential', 9).mpl_colors[::-1],
    'brewer_div_4': brewer2mpl.get_map('PuOr', 'Diverging', 11).mpl_colors[::-1] + brewer2mpl.get_map('Greys', 'sequential', 9).mpl_colors[::-1],
    'brewer_div_5': brewer2mpl.get_map('RdBu', 'Diverging', 11).mpl_colors[::-1] + brewer2mpl.get_map('Greys', 'sequential', 9).mpl_colors[::-1],
}

COLOR = COLORSETS['brewer_div_2']

PARAMS = {
       'axes.labelsize': 8,
       'text.fontsize': 8,
       'legend.fontsize': 10,
       'legend.fancybox': False,
       'xtick.labelsize': 10,
       'ytick.labelsize': 10,
       'figure.dpi': 300,
       'savefig.dpi': 300,
       'text.usetex': False,
       }
plt.rcParams.update(PARAMS)

CSETS ={
    'gray_scale': {'name': 'Greys', 'type': 'Sequential', 'max_colors': 9, 'reverse':True}, # 10

    'spectral':{'name': 'Spectral', 'type': 'Diverging', 'max_colors': 11, 'reverse':True},          # 5
    'divergent_2':{'name': 'BrBg', 'type': 'Diverging', 'max_colors': 11, 'reverse':True},         # 8 Spectral
    'divergent_3':{'name': 'RdYlGn', 'type': 'Diverging', 'max_colors': 11, 'reverse':True},          # 7
    'divergent_4':{'name': 'RdYlBu', 'type': 'Diverging', 'max_colors': 11, 'reverse':True},           # 5

    'qualitative_1':{'name': 'Set1', 'type': 'qualitative', 'max_colors': 9, 'reverse':False},        # 5 Pastel
    'qualitative_2':{'name': 'Dark2', 'type': 'qualitative', 'max_colors': 8, 'reverse':False},        # 5
    'qualitative_3':{'name': 'Set2', 'type': 'qualitative', 'max_colors': 8, 'reverse':False},        # 5 Pastel
    'qualitative_4':{'name': 'Set3', 'type': 'qualitative', 'max_colors': 12, 'reverse':False},        # 5 Pastel

    'c3':{'name': 'c3', 'type': 'c3', 'max_colors': 10, 'reverse': False}

}

C3_COLORS=[
    (53/255, 132/255, 187/255),
    (255/255, 140/255, 38/255),
    (65/255, 169/255, 65/255),
    (218/255, 61/255, 61/255),
    (158/255, 118/255, 195/255),
    (151/255, 103/255, 93/255),
    (229/255, 132/255, 200/255),
    (140/255, 140/255, 140/255),
    (194/255, 195/255, 56/255),
    (46/255, 196/255, 211/255)
]


def get_colorset(set_name, n):

    if set_name == 'c3':
        return C3_COLORS[:n]
    else:
        return brewer2mpl.get_map(CSETS[set_name]['name'], CSETS[set_name]['type'], n).mpl_colors[:n]

def get_color(set_name, n, background_color):
    '''

    :param: background_color is hex color
    '''

    background_color = background_color.lstrip('#')
    background_color_rgb = tuple(int(background_color[i:i+2], 16)/255 for i in (0, 2,4))

    set = CSETS[set_name]

    if n <= set['max_colors']:
        try:
            colors =  get_colorset(set_name, n)
        # many sets have minimum ns, so update n when value error
        except ValueError:
            colors =  get_colorset(set_name, set['max_colors'])
    else:
        colors = (n//set['max_colors']+1) * get_colorset(set_name, CSETS[set_name]['max_colors'])

    if not set['reverse']:
        colors = colors[::-1]

    # remove colors that are similar to the background color
    final_colors = []
    for color in colors:
        dif = (color[0]-background_color_rgb[0])**2 +  (color[1]-background_color_rgb[1])**2 +  (color[2]-background_color_rgb[2])**2
        if dif > 0.15:
            final_colors.append(color)



    if len(final_colors) < n:
        final_colors = []
        colors = (n//set['max_colors']+1)*2 * get_colorset(set_name, CSETS[set_name]['max_colors'])
        if not set['reverse']:
            colors = colors[::-1]
        for color in colors:
            dif = (color[0]-background_color_rgb[0])**2 +  (color[1]-background_color_rgb[1])**2 +  (color[2]-background_color_rgb[2])**2
            if dif > 0.15:
                final_colors.append(color)

    final_colors = final_colors[-n:]
    if n == 1:
        final_colors = final_colors[0]


    return final_colors


#c = get_color('gray_scale', 9, '#f0f0f0')


def create_title_and_subtitle(display_type, data_type, search_tokens):
    '''
    Frequencies of addiction across collections
    '''


    display_type_dict = {
        'frequencies':'Frequencies',
        'counts': 'Counts',
        'z_scores': 'Z-Scores'}

    data_type_dict = {
        'collections': 'collections',
        'doc_types': 'document types',
        'doc_type_groups': 'document types'
    }


    title = "{} of {}".format(display_type_dict[display_type], format_list_for_string(search_tokens))

    if data_type in ['collections', 'doc_types', 'doc_type_groups']:
        title += ' across {}'.format(data_type_dict[data_type])

    subtitle = ''
    return title, subtitle




def format_list_for_string(list):

    if len(list) == 1:
        return list[0]
    elif len(list) == 2:
        return " and ".join(list)
    else:
        return ", ".join(list[:-2] + [", and ".join(list[-2:])])

def plot_frequencies(
        search_tokens, doc_type_filters, collection_filters, availability_filters, term_filters,
        display_type='counts', data_type='collections', stacked='stacked', smoothing=1,
        png_width=8, png_height=5,
        start_year=1901, end_year=2016,

        line_width=2, pad_inches=0.3,

        colorset = 'spectral',

        title = '', subtitle='',

        chart_font_size=9, title_font_size=16, subtitle_font_size=12,

        legend_position='in_top_right', legend_show_frame=True,
        legend_show_counts=False, legend_show_freqs_or_z=False, legend_show_heading=True,
        background_color='#f0f0f0', grid_color='#cbcbcb', frame_color='#4d4d4d', text_color='#000000',

        excluded_tokens = '["Philip Morris"]',
        output_to_web=False
    ):

    plt.close('all')

    dpi = plt.rcParams['figure.dpi']

    PARAMS = {
        'axes.facecolor': background_color,
        'axes.edgecolor': background_color,
        'savefig.edgecolor': background_color,
        'savefig.facecolor': background_color,
        'grid.color': grid_color,

       }
    plt.rcParams.update(PARAMS)


    if not chart_font_size:
        chart_font_size = png_width+2
    if not title_font_size:
        title_font_size = png_width*2
    if not subtitle_font_size:
        subtitle_font_size = png_width*4/3
    if not line_width:
        line_width = png_width/3.5

    excluded_tokens = json.loads(excluded_tokens)

    data = get_frequencies_results(search_tokens, doc_type_filters, collection_filters, availability_filters, term_filters)
    totals_for_sorting = [sum(entry['counts'][start_year-1901:end_year+1-1901]) for entry in data['data'][data_type]]

    search_tokens = [token['token'] for token in data['data']['tokens']]

    if not title:
        title, _ = create_title_and_subtitle(display_type, data_type, search_tokens)


#    data = [entry for _, entry in sorted(zip(totals_for_sorting, data['data'][data_type]))][::-1]
    data = [entry for _, entry in sorted(zip(totals_for_sorting, data['data'][data_type]))][::-1]
    display_tokens_orig = [entry['token'].strip() for entry in data]

    # exclude data if necessary
    if len(excluded_tokens) > 0:
        data_with_exclusions = []
        for entry in data:
            if not entry['token'].strip() in excluded_tokens: # for some reason, the token includes a spcae at the end
                data_with_exclusions.append(entry)
        data = data_with_exclusions


    display_data = [moving_avg(entry[display_type][start_year-1901:end_year+1-1901], smoothing) for entry in data]
    display_tokens = [entry['token'].strip() for entry in data]


    year = np.array([i for i in range(start_year, end_year+1)])
    year_interpolated = np.linspace(start_year, end_year, 1000)
    data_interpolated = [
        np.clip(spline(year, np.array(token), year_interpolated), 0, np.inf) for token in display_data
    ]

    no_tokens = len(display_tokens)

    totals = [sum(d['counts'][start_year-1901:end_year+1-1901]) for d in data]
    longest_total = max([len("{:,}".format(total)) for total in totals])
    frequencies = [d['frequencies'][start_year-1901:end_year+1-1901] for d in data]
    mean_frequencies = [np.mean(f) for f in frequencies]
    if display_type == 'z_scores':
        z_scores = [d['z_scores'][start_year-1901:end_year+1-1901] for d in data]
        mean_z_scores = [np.mean(z) for z in z_scores]



    d = {display_tokens[i]:data_interpolated[i] for i in range(len(display_tokens))}
    d['Year'] = year_interpolated
    df = DataFrame(d)


    if stacked == 'stacked':

        # fig = plt.figure()
        # graph = plt.subplot2grid((1,1), (0,0))
        # graph.stackplot(df['Year'], display_data)
        alpha = 0.25
        if colorset == 'gray_scale':
            alpha = 1
        graph = df.plot.area(x='Year',
                    y=display_tokens[::-1],
                    figsize=(png_width, png_height),
                    linewidth=line_width/2,
                    alpha=alpha,
                    color=get_color(colorset, no_tokens, background_color),
                    )


    else:
        graph = df.plot(x='Year',
                    y=display_tokens,
                    figsize=(png_width, png_height),
                    linewidth=line_width,
                    color=get_color(colorset, no_tokens, background_color)
                    )

    '''
    Axes
    '''

    graph.tick_params(axis = 'both', which = 'major', labelsize = chart_font_size)
    # add additional year left and right
    graph.set_xlim(left=start_year-1, right=end_year)
    # hide "Year" in graph
    graph.xaxis.label.set_visible(False)
    # move axes labels away from graph
    graph.tick_params(axis='x', pad=6, colors=text_color)
    graph.tick_params(axis='y', pad=6, colors=text_color)
    graph.locator_params(axis='y', nbins=5)


    if display_type == 'frequencies':
        graph.yaxis.set_major_formatter(tick.FuncFormatter(y_formatter))


    '''
    Border
    '''

    for spine in ['bottom', 'top', 'right', 'left']:
        graph.spines[spine].set_color(frame_color)
        graph.spines[spine].set_linewidth(png_width/10)

    '''
    Legend
    '''

    if legend_position in ['out_top_left', 'out_top_right', 'in_top_right', 'in_top_left']:

        handles, _ = graph.get_legend_handles_labels()

        # make handles alpha = 1 (can't do directly as that resets the chart to alpha =1)
        if stacked == 'stacked':
            colors = get_color(colorset, no_tokens, background_color)
            if no_tokens == 1: colors = [colors]
            new_handles = []
            for idx, handle in enumerate(handles):
                new_handles.append(Rectangle((0, 0), 0.5, 0.5, fc=colors[idx], fill=True, edgecolor='none', linewidth=0, alpha=1))
            handles=new_handles

        extra = Rectangle((0, 0), 0.5, 0.5, fc="w", fill=False, edgecolor='none', linewidth=0)
        legend_no_col = 2

        if legend_show_counts or legend_show_freqs_or_z:
            legend_show_heading = True

        heading_dict={'tokens': 'Term', 'collections':'Collection', 'doc_types': 'Document Types', 'doc_type_groups': 'Document Type Groups'}
        if legend_show_heading:
            handles = [extra] + handles[::-1] + [extra]*(no_tokens+1)
            labels = [''] * (no_tokens+1) + [heading_dict[data_type]] + display_tokens
        else:
            handles = handles[::-1] + [extra] * no_tokens
            labels = [''] * (no_tokens) + display_tokens

        if legend_show_counts:
            legend_no_col += 1
            handles += [extra] * (no_tokens+1)
            labels += ['Total'] + ["{:,}".format(total) for total in totals]

        if legend_show_freqs_or_z:
            legend_no_col += 1
            handles += [extra] * (no_tokens+1)
            if display_type == 'frequencies' or display_type == 'counts':
                labels += ['Mean Frequency'] + ["{:1.3f}%".format(round(f*100, 4)) for f in mean_frequencies]
            elif display_type == 'z_scores':
                labels += ['Mean Z-Score'] + ["{:1.3f}".format(round(f, 4)) for f in mean_z_scores]


        handle_text_pad = -1
        if legend_show_counts or legend_show_freqs_or_z:
            handle_text_pad = -2.3

        if legend_position == 'out_top_right':
            graph.legend(handles, labels,
                         ncol=legend_no_col,
                         prop={'family': 'DejaVu Sans', 'size': chart_font_size},
                         fontsize=chart_font_size,
                         handletextpad = handle_text_pad,
                         loc=2,
                         bbox_to_anchor=(1.02, 1),
                         framealpha=1,
                         borderaxespad=0,
                         borderpad=1.5
            )
        if legend_position == 'out_top_left':
            graph.legend(handles, labels,
                         ncol=legend_no_col,
                         prop={'family': 'DejaVu Sans', 'size': chart_font_size},
                         fontsize=chart_font_size,
                         handletextpad = handle_text_pad,
                         loc=1,
                         bbox_to_anchor=(-0.1, 1),
                         framealpha=1,
                         borderaxespad=0,
                         borderpad=1.5

            )
        elif legend_position == 'in_top_right':
            graph.legend(handles, labels,
                         ncol=legend_no_col,
                         prop={'family': 'DejaVu Sans', 'size': chart_font_size},
                         fontsize=chart_font_size,
                         handletextpad = handle_text_pad,
                         loc=1,
                         bbox_to_anchor=(1.00,1),
                         framealpha=1,
                         borderaxespad=0,
                         borderpad=1
            )
        elif legend_position == 'in_top_left':
            graph.legend(handles, labels,
                         ncol=legend_no_col,
                         prop={'family': 'DejaVu Sans', 'size': chart_font_size, },
                         fontsize=chart_font_size,
                         handletextpad = handle_text_pad,
                         loc=2,
                         bbox_to_anchor=(-0.0,1),
                         framealpha=1,
                         borderaxespad=0,
                         borderpad=1
            )



        for idx, t in enumerate(graph.get_legend().get_texts()):
            t.set_color(text_color)

            # heading bold
            if legend_show_heading and idx in [no_tokens+1, no_tokens*2+2, no_tokens*3+3]:
                text = t.get_text()
                new_label = "".join([i + r"\/" for i in text.split()])
                t.set_text(r'$\mathbf{}$'.format("{" + new_label + "}"))
#                t.set_text(r'$\mathrm{\mathsf{\mathbf{sansserif}}}$')

            # right align counts and frequencies/z-scores
            if idx >= no_tokens*2+2:
                t.set_ha('right')

            # align tokens
            if legend_show_heading and idx >= no_tokens*1+1 and idx < no_tokens*2+2:
                if legend_show_counts or legend_show_freqs_or_z:
                    t.set_x(chart_font_size*1.3*0.01*dpi)
                else:
                    # if only tokens, don't move them.
                    t.set_x(0)


            # align top token if heading not shown
#            if not legend_show_heading and idx == no_tokens:
#                t.set_x(chart_font_size*1.3*0.01*dpi)

            # align counts
            if idx >= no_tokens*2+2 and idx < no_tokens*3+3:
                t.set_x(chart_font_size*2.5*0.01*dpi + chart_font_size*longest_total/2*0.01*dpi)

            # align frequencies/z-scores
            if idx >= no_tokens*3+3:
                t.set_x(chart_font_size*8*0.01*dpi + chart_font_size*longest_total/2*0.01*dpi)

    # legend frame
    if legend_show_frame:
        graph.get_legend().get_frame().set_linewidth(png_width/10)
        graph.get_legend().get_frame().set_edgecolor(frame_color)
    else:
        graph.get_legend().set_frame_on(False)


    canvas = graph.figure.canvas
    fig = graph.get_figure()



    '''
    Title and subtitle
    '''
    subtitle_plot = graph.set_title(subtitle, fontsize=subtitle_font_size, horizontalalignment='left',
                    x=0)
    subtitle_plot.draw(canvas.get_renderer())

    subtitle_bbox = subtitle_plot.get_window_extent().get_points()

    if subtitle:
        title_plot = graph.annotate(xy=(0, subtitle_bbox[1][1]-png_height/20*dpi) ,xycoords='axes pixels', color=text_color,
                                    s=title, weight="bold", horizontalalignment='left', fontsize=title_font_size, verticalalignment='bottom')
    else:
        title_plot = graph.annotate(xy=(-0.000,1.04), xycoords='axes fraction', color= text_color,
                               s=title, weight="bold", horizontalalignment='left',
                                fontsize=title_font_size, verticalalignment='bottom')

    # set line alpha to 1
    if stacked=='stacked':
        for l in fig.gca().lines:
            l.set_alpha(1)

    if output_to_web:
        return fig, display_tokens_orig

    else:
        fig.savefig('out.png', bbox_inches='tight', pad_inches = pad_inches)
        print("fig1", fig)
        plt.close(fig)
        plt.close('all')
        print("fig2", fig)
        gc.collect()
        fig = plt.gcf()
        plt.close(fig)
        print("fig3", fig)

        gc.collect()

        del data_interpolated
        del year_interpolated

        gc.collect()





def y_formatter(x,y):

    return "{:1.4f}%".format(round(x*100, 4))

def moving_avg(data, smoothing):



    if smoothing == 0: return data

    moving_avg = len(data) * [0]

    for idx in range(len(data)):
        section = data[max(0, idx - smoothing): idx+smoothing+1]
        moving_avg[idx] = np.sum(section)/len(section)

    return moving_avg



if __name__ == "__main__":

    plot_frequencies('inbifo', '[]', '[]', '[]', '', output_to_web=False)