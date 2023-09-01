import filepattern as fp
import pprint
from bfio import BioReader, BioWriter
import numpy as np
from math import ceil
from pathlib import Path

def main():
    # STITCH_PATH = Path("/Users/antoinegerardin/Documents/projects/polus-plugins/transforms/images/image-assembler-plugin/data/nist_mist_dataset/Small_Phase_Test_Dataset_Example_Results/img-global-positions-0.txt")
    STITCH_PATH = Path("/Users/antoinegerardin/Documents/data/image-assembler-plugin/nist-mist-dataset/vectors/img-global-positions-0.txt")

    # IMG_PATH = Path("/Users/antoinegerardin/Documents/projects/polus-plugins/transforms/images/image-assembler-plugin/data/nist_mist_dataset/Small_Phase_Test_Dataset/image-tiles")
    IMG_PATH = Path("/Users/antoinegerardin/Documents/data/image-assembler-plugin/nist-mist-dataset/images")
    pattern = fp.infer_pattern(STITCH_PATH)

    OUTPUT_PATH = Path("examples/data") / "out.ome.tiff"

    fovs = fp.FilePattern(STITCH_PATH, pattern)

    # guess a name for the final assembled image
    output_name = fovs.output_name()

    print(pattern)
    print(output_name)

    # that will decide how much work we do before 
    # writing to disk
    # TODO this should be benchmarked, maybe made a param
    chunk_width = 2048
    chunk_height = 2048

    # for fov in fovs():
    #     pprint.pprint(fov)

    # let's figure out the size of a partial FOV.
    # Pick the first image in stitching vector.
    # We assume all images have the same size.
    first_image_name = fovs[0][1][0]
    first_image = IMG_PATH / first_image_name

    # stitching is only perform on a x,y plane, 
    # so we can safely assume dim(first_image == 2)
    fov_width, fov_height = size = BioReader.image_size(first_image)
    print(f"fov size : {size}")

    with BioReader(first_image) as br:
     full_image_metadata = br.metadata
     size = br.image_size
     fov_width = br.x
     fov_height = br.y

     # TODO CHECK if there is a need to assemble 3D fovs.
     # It is unlikely given that some stitching 
     # and alignment would needed to occur beforehand.

    # find final image size
    full_image_width, full_image_height = fov_width, fov_height
    for fov in fovs():
        metadata = fov[0]
        full_image_width = max(full_image_width, metadata['posX'] + fov_width)
        full_image_height = max(full_image_height, metadata['posY'] + fov_height)

        
    print("full image size: ", full_image_width, full_image_height)

    # number of chunks we need to write in the x,y plane
    chunk_grid_col = ceil(full_image_width / chunk_width)
    chunk_grid_row = ceil(full_image_height / chunk_height)
    chunk_grid_size = (chunk_grid_col, chunk_grid_row)

    chunks = [[[] for col in range(chunk_grid_col)] for row in range(chunk_grid_row)]

    region_count = 0

    # figure out regions of fovs that needs to be copied into each chunk.
    # This is fast so it can be done beforehand in a single process.
    for fov in fovs():
        # we are parsing a stitching vector, so we are always getting unique records.        
        assert(len(fov[1]) == 1)
        filename = fov[1][0]

        # get global coordinates of fov
        metadata = fov[0]
        global_fov_start_x = metadata['posX']
        global_fov_start_y = metadata['posY']

        # check which chunks the fov overlaps
        chunk_col_min = global_fov_start_x // chunk_width
        chunk_col_max = (global_fov_start_x + fov_width) // chunk_width
        chunk_row_min = global_fov_start_y // chunk_height
        chunk_row_max = (global_fov_start_y + fov_height)  // chunk_height

        max_end_x = 0
        max_end_y = 0

        # define fov's contributions to each chunk as regions
        for row in range(chunk_row_min, chunk_row_max + 1):
            for col in range(chunk_col_min, chunk_col_max + 1):
                print("row, col : ", row, col)
                # global coordinates of the contribution
                global_start_x = max(global_fov_start_x, col * chunk_width)
                global_end_x = min(global_fov_start_x + fov_width, (col + 1) * chunk_width)
                global_start_y = max( global_fov_start_y, row * chunk_height)
                global_end_y = min(global_fov_start_y + fov_height, (row + 1) * chunk_height)

                print(f"{global_start_x}->{global_end_x},{global_start_y}->{global_end_y}") # remove

                assert(global_start_x >= col * chunk_width)
                assert(global_start_x >= global_fov_start_x )
                assert(global_start_x <= (col + 1) * chunk_width)
                assert(global_start_y >= row * chunk_height)
                assert(global_start_y <= (row + 1) * chunk_height)

                # local coordinates within the fov itself
                fov_start_x = max(global_fov_start_x, col * chunk_width) - global_fov_start_x
                fov_start_y = max(global_fov_start_y, row * chunk_height) - global_fov_start_y

                region = (
                        filename, 
                        (global_start_x, global_end_x, global_start_y, global_end_y), 
                        (fov_start_x, fov_start_y)
                        )
                
                print(f"region : {region}")

                chunks[row][col].append(region)

                # TODO remove
                region_count = region_count + 1
                max_end_x = max (max_end_x, global_end_x)
                max_end_y = max (max_end_y, global_end_y)

    # TODO remove
    # assert max_end_x == 5936 , f"max_x {max_end_x} should 5936"
    # assert max_end_y == 4453, f"max_y {max_end_y} should 4453" #TODO CHECK WHY!

    print("# of regions: ", region_count)

    # TODO CHECK For now we have a single writer.
    # Check implementation of bfio to see if this makes sense.
    with BioWriter(OUTPUT_PATH, 
                   metadata=full_image_metadata, 
                   backend="python") as bw:
        bw.x =  full_image_width
        bw.y = full_image_height
        bw._CHUNK_SIZE = 2048 # should match our chunk size

        # now copy each fov regions.
        # This requires multiple reads and copies and a final write.
        # This is a slow IObound process so it can benefit multithreading
        # TODO add multithreading.
        for row in range(chunk_grid_row):
            for col in range(chunk_grid_col):

                chunk = np.zeros((chunk_width, chunk_height)) # TODO remove,we use bfio supertile

                for region in chunks[row][col]:
                    print(f"process region : {region} for chunk ({col},{row})")
                    filepath = IMG_PATH / region[0]
                    with BioReader(filepath) as br:
                        (global_start_x, global_end_x, global_start_y, global_end_y) = region[1]
                        (fov_start_x, fov_start_y) = region[2]

                        region_width = global_end_x - global_start_x
                        region_height = global_end_y - global_start_y

                        print(f"copying region from fov : {fov_start_x, fov_start_y} with width ({region_width}, height {region_height})")

                        data = br[fov_start_y: fov_start_y + region_height ,fov_start_x:fov_start_x + region_width]

                        chunk_start_x = global_start_x - col * chunk_width
                        chunk_start_y = global_start_y - row * chunk_height
                        chunk_end_x = chunk_start_x + region_width
                        chunk_end_y = chunk_start_y + region_height

                        print(f"to chunk {chunk_start_x}->{chunk_end_x} , {chunk_start_y}->{chunk_end_y}")
                        chunk[chunk_start_y:chunk_end_y, chunk_start_x:chunk_end_x] = data
        
        max_x = min((row + 1) * chunk_height, full_image_height)
        max_y = min((col + 1) * chunk_width, full_image_width)

        print(bw.shape)
        print(chunk.shape)
        chunk = chunk[..., np.newaxis, np.newaxis, np.newaxis]
        print(chunk.shape)

        print(max_x - row * chunk_height)
        print(max_y - col * chunk_width)

        bw[row * chunk_height:max_x, (col * chunk_width):max_y] = chunk

if __name__ == "__main__":
    main()

# for file in files(): 
#     pprint.pprint(file)

# filenames = [f for f in files()]

# for filename in filenames:
#     pprint(filename)

# files(group_by='c')

# unique_values = files.get_unique_values()

# print(unique_values)



# print(pattern)