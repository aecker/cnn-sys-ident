import numpy as np
from numpy import pi


class GaborSet:
    def __init__(self,
                 canvas_size,  # width x height
                 center_range, # [x_start, x_end, y_start, y_end]
                 sizes, # +/- 2 SD of envelope
                 spatial_frequencies,  # cycles / envelop SD, i.e. depends on size
                 contrasts,
                 orientations,
                 phases,
                 relative_sf=True):   # scale SF by size (True) or use absolute units (False)
        self.canvas_size = canvas_size
        cr = center_range
        self.locations = np.array(
            [[x, y] for x in range(cr[0], cr[1]) 
                    for y in range(cr[2], cr[3])])
        self.sizes = sizes
        self.spatial_frequencies = spatial_frequencies
        self.contrasts = contrasts
        if type(orientations) is not list:
            self.orientations = np.arange(orientations) * pi / orientations
        else:
            self.orientations = orientations
        if type(phases) is not list:
            self.phases = np.arange(phases) * (2*pi) / phases
        else:
            self.phases = phases
        self.num_params = [
            self.locations.shape[0],
            len(sizes),
            len(spatial_frequencies),
            len(contrasts),
            len(self.orientations),
            len(self.phases),
        ]
        self.relative_sf = relative_sf

    def params_from_idx(self, idx):
        c = np.unravel_index(idx, self.num_params)
        location = self.locations[c[0]]
        size = self.sizes[c[1]]
        spatial_frequency = self.spatial_frequencies[c[2]]
        if self.relative_sf:
            spatial_frequency /= size
        contrast = self.contrasts[c[3]]
        orientation = self.orientations[c[4]]
        phase = self.phases[c[5]]
        return location, size, spatial_frequency, contrast, orientation, phase
        
    def params_dict_from_idx(self, idx):
        (location, size, spatial_frequency, 
            contrast, orientation, phase) = self.params_from_idx(idx)
        return {
            'location': location,
            'size': size,
            'spatial_frequency': spatial_frequency,
            'contrast': contrast,
            'orientation': orientation,
            'phase': phase,
        }

    def gabor_from_idx(self, idx):
        return self.gabor(*self.params_from_idx(idx))

    def gabor(self, location, size, spatial_frequency, contrast, orientation, phase):
        x, y = np.meshgrid(np.arange(self.canvas_size[0]) - location[0],
                           np.arange(self.canvas_size[1]) - location[1])
        R = np.array([[np.cos(orientation), -np.sin(orientation)],
                      [np.sin(orientation),  np.cos(orientation)]])
        coords = np.stack([x.flatten(), y.flatten()])
        x, y = R.dot(coords).reshape((2, ) + x.shape)
        envelope = 0.5 * contrast * np.exp(-(x ** 2 + y ** 2) / (2 * (size/4)**2))
        
        grating = np.cos(spatial_frequency * x * (2*pi) + phase)
        return envelope * grating

    def image_batches(self, batch_size):
        num_stims = np.prod(self.num_params)
        for batch_start in np.arange(0, num_stims, batch_size):
            batch_end = np.minimum(batch_start + batch_size, num_stims)
            images = [self.gabor_from_idx(i)
                          for i in range(batch_start, batch_end)]
            yield np.array(images)

    def images(self):
        num_stims = np.prod(self.num_params)
        return np.array([self.gabor_from_idx(i) for i in range(num_stims)])
