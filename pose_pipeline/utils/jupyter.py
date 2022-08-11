from IPython.display import Video, HTML, display


def play(video, width=640):
    video_kwargs = {"width": width, "html_attributes": "controls autoplay loop"}
    video = video.fetch1("output_video")
    display(Video(video, **video_kwargs))


def play_grid(videos, height=200):
    # nicely handle a single row
    if not isinstance(videos[0], list):
        videos = [videos]

    video_kwargs = {"height": height, "html_attributes": "controls autoplay loop"}

    # get the videos and their HTML embedding
    videos = [
        [Video(v.fetch1("output_video"), **video_kwargs)._repr_html_() if v is not None else "" for v in vrow]
        for vrow in videos
    ]

    # build a row up
    display(
        HTML(
            "<table><tr>{}</tr></table>".format(
                "</tr><tr>".join("<td>{}</td>".format("</td><td>".join(str(_) for _ in row)) for row in videos)
            )
        )
    )
