import mmcensor.const as mmc

def sizes_to_key( sizes ):
    key = 0
    supported = mmc.supported_sizes
    for i in range(len(supported)):
        if supported[i] in sizes:
            key = key + 2**i

    return key

def key_to_sizes( key ):
    sizes = []
    supported = mmc.supported_sizes
    for i in range(len(supported)):
        if key&2**i:
            sizes.append( supported[i] )
    return sizes
