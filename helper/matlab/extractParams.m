%matlab code that call function and save the parameters
function [] = extractParams()

    % params

    funcPath = "/home/matmat/Documents/urban_lab/002_Superresolution/raw_data_mat/005_mouse/";
    rfMatPath = "/home/matmat/Documents/urban_lab/002_Superresolution/raw_data_mat/005_mouse/001_RF.mat";

    jsonOutputPath = "/home/matmat/Documents/urban_lab/002_Superresolution/superresolution/data_bf/mouse/params.json";

    addpath(funcPath);

    rfMat = load(rfMatPath);
    rfMatField = fieldnames(rfMat);

    params = Copy_of_paramLinear15M_Dax();

    % disp(params.bmode);
    % disp(params.acq);
    % disp(params.hard);
    % disp(params.probe);
    % disp(params.bf);
    % disp(params.filter);

    array = struct();

    array.length = params.probe.nx;
    array.aperture = params.bf.fd;
    array.pitch = params.probe.dx * 10^-3;
    array.acq_depth = [params.acq.z(1) * 10^-3, params.acq.z(2) * 10^-3];

    disp(array)

    rf = struct();

    rf.frequency = params.acq.freqTX * 10^6;
    rf.samples = size(rfMat.(rfMatField{1}), 1); 
    rf.sampling_rate = params.acq.samplesPerWave * rf.frequency;
    rf.C = params.acq.c * 10^3;
    rf.angles = params.bmode.anglex;
    rf.delay0 = params.acq.z(1) * 10^-3;
    rf.interpolation_factor = 2;

    disp(rf)

    image = struct();

    image.x0 = params.bf.x0 * 10^-3;
    image.z0 = rf.delay0;    %in m
    image.dx = params.bf.dx * 10^-3;
    image.dz = params.bf.dz * 10^-3;
    image.nx = params.bf.nx;
    image.nz = round(((params.acq.z(2) * 10^-3) - (params.acq.z(1) * 10^-3)) / image.dz);
    image.nframes = 1;

    disp(image)

    encoded = jsonencode(struct('array', array, 'rf', rf, 'image', image));

    fid = fopen(jsonOutputPath,'w');
    fprintf(fid,'%s',encoded);
    fclose(fid);

end