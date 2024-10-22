# KROME 1.3.1

[![LinkedIn][linkedin-shield]][linkedin-url]

- Python package created as a Research Associate at UAE University. The package contains modules that calculate magnetic activity indices from spectroscopic data of M dwarfs. 
- It accepts file types of .s taken from the spectropolarimeter __NARVAL__ & __ESPADONS__ and ADP.fits/s1d_A.fits from the spectrographs __HARPS__,__HARPS-N__,__SOPHIE.__ and __ELODIE__.

## Installation
- The package can be downloaded using `pip` as `pip install krome`

## Usage
- Follow the [Tutorial](https://github.com/MXK606/krome/blob/main/Tutorial.ipynb) which explains each activity index method in detail.

## Code Dependencies 
- numpy
- matplotlib
- pandas
- tqdm
- astropy
- PyAstronomy
- specutils
- uncertainties

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/kmukul

## Citing

- If you use this software in your research, please cite the article "A Study of the Magnetic Activity and Variability of GJ 436" in the MNRAS journal using the following BibTex entry:

```
@ARTICLE{2023MNRAS.518.3147K,
       author = {{Kumar}, M. and {Fares}, R.},
        title = "{A study of the magnetic activity and variability of GJ 436}",
      journal = {\mnras},
     keywords = {stars: activity, stars: chromospheres, stars: individual: GJ 436, planetary systems, Astrophysics - Solar and Stellar Astrophysics},
         year = 2023,
        month = jan,
       volume = {518},
       number = {2},
        pages = {3147-3163},
          doi = {10.1093/mnras/stac276610.48550/arXiv.2209.11258},
archivePrefix = {arXiv},
       eprint = {2209.11258},
 primaryClass = {astro-ph.SR},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023MNRAS.518.3147K},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```


