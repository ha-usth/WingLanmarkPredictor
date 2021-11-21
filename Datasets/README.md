
We evaluated the prediction accuracy of our framework with 5 public
insect-wing datasets summarized by Table [1](#tab:dataset) in which the
"Short name" is entitled for the sake of mentioning afterward. Fig.
[1](#fig:ds) provides the illustration for examples and landmarks of
these datasets. Their details are discussed as follows:

  - *Droso-small* was employed to test the method in paper . For
    comparison with this state of the art method, we conducted
    thoroughly benchmarks on both accuracy and speed.

  - *Droso-big* is another dataset of Drosophila melanogaster wings
    introduced in . Among insect species, drosophila can be encountered
    in numerous public datasets as it has drawn significant attention in
    the studies of automatic wing shape analysis . This dataset was
    published with the target to be the general resource for community.
    Accordingly, it contains much larger amount of wing images (1134
    images for each side) than the other four. Its genotype diversity
    leading to the variation of landmark shape and location that may
    challenge our framework. To prepare for future comparison tasks
    between the 2 datasets on drosophila, we annotated the landmark
    classes of *Droso-big* as in *Droso-small*. Since the left and right
    wings of organisms are almost symmetrical, we made annotation for
    right side solely to save our effort.

  - *Fly* is a small dataset  of Glossina palpalis in southern Ivory
    Coast. The small distance between some landmark classes pose a risk
    of confusions in landmark identity.

  - *Bactro* dataset  comprises 53 images of both male and female
    bactrocera tau from Kanchanaburi and Nan region in Thailand. This
    diversity of geographic origins and sexes may result in variations
    in morphology and locations of landmarks. Moreover, some of landmark
    classes fall into the opaque areas of wing leading to low contrast
    levels which is probably a difficulty for detection task.

  - *Diacha* is a dataset of Diachasmimorpha longicaudata , a genus of
    endoparasitoid. The obstacles for landmark dection methods are the
    existences of stains, water drops and wet areas on wings.

<div id="tab:dataset">

| Short name     | Species                      | Number of landmark classes | Dataset size | Resolution |  |
| :------------- | :--------------------------- | :------------------------- | :----------- | :--------- | :- |
| *Droso-small * | Drosophila                   | 15                         | 138          | 1400x900   |  |
| *Droso-big *   | Drosophila                   | 15                         | 1134         | 1360x1024  |  |
| *Fly *         | Tsetse fly                   | 10                         | 15           | 2031x1180  |  |
| *Bactro *      | Bactrocera tau               | 12                         | 53           | 2048x1563  |  |
| *Diacha *      | Diachasmimorpha longicaudata | 10                         | 92           | 1000x750   |  |

<span id="tab:dataset" label="tab:dataset">\[tab:dataset\]</span>
Summary of the <span style="color: red">experiment</span> datasets.

</div>

![Sample images and the corresponding landmarks of 5 tested datasets.
From left to right side, upper to lower: Droso-small, Droso-big, Fly,
Bactro, Diacha.](img/ds.png)
