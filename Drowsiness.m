clc;
clear all;
close all;
%%
load DB
load svm
cl = {'open','close'};

dim = [30 60;
        30 60
        40 65];
%% start(vid);
delete(imaqfind)
vobj=videoinput('winvideo',1);
triggerconfig(vobj,'manual');
set(vobj,'FramesPerTrigger',1 );
set(vobj,'TriggerRepeat', Inf);

%%  Viewing the default colour space used for the data
colour_spec=vobj.ReturnedColorSpace;

%% Modifying the colour space used for the data
if  ~strcmp(colour_spec,'rgb')
    set(vobj,'ReturnedColorSpace','rgb');
end

start(vobj)

%% object detection
FaceDetector = vision.CascadeObjectDetector;  
FaceDetectorLeye = vision.CascadeObjectDetector('EyePairBig');
FaceDetectorMouth = vision.CascadeObjectDetector('Mouth');
tic

%% Initialise vector
LC = 0;
RC = 0;
MC = 0;
TF = 0;
TC = 0;
Feature = [];
c1p = 1;
species = 'Non-Fatigue';
%%
for ii = 1:600
   
    trigger(vobj);
    im=getdata(vobj,1); % Gets the frame in im
    imshow(im)
   
    subplot(2,2,1);
    imshow(im)
   
    % Detect faces
    bbox = step(FaceDetector, im);
   
    if ~isempty(bbox);
        bbox = bbox(1,:);

        % Plot box
        rectangle('Position',bbox,'edgecolor','r');

         S = skin_seg2(im);
   
        % Segment skin region
        bw3 = cat(3,S,S,S);

        % Multiply with original image and show the output
        Iss = double(im).*bw3;

        Ic = imcrop(im,bbox);
        Ic1 = imcrop(Iss,bbox);
        subplot(2,2,2);
        imshow(Ic)

        
       %% detects eyes
        bbox_eye = step(FaceDetectorLeye, Ic);
       
        if ~isempty(bbox_eye);
            bbox_eye = bbox_eye(1,:);

            E_eye = imcrop(Ic,bbox_eye);
            % Plot box
            rectangle('Position',bbox_eye,'edgecolor','y');
        else
            disp('Eyes not detected')
        end
       
        if isempty(bbox_eye)
            continue;
        end
       Ic(1:bbox_eye(2)+2*bbox_eye(4),:,:) = 0;

        %% Detect Mouth
        bbox_M = step(FaceDetectorMouth, Ic);
       

        if ~isempty(bbox_M);
            bbox_M_temp = bbox_M;
           
            if ~isempty(bbox_M_temp)
           
                bbox_M = bbox_M_temp(1,:);
                E_mouth =  imcrop(Ic,bbox_M);

                % Plot box
                rectangle('Position',bbox_M,'edgecolor','y');
            else
                            continue;
            end
        else
                     continue;
        end
       
        [nre nce k ] = size(E_eye);
       
        %% Divide into two parts
        Left_eye = E_eye(:,1:round(nce/2),:);
        Right_eye = E_eye(:,round(nce/2+1):end,:);
          
         
        E_mouth_3 = E_mouth;
       
       
        Left_eye = rgb2gray(Left_eye);
        Right_eye = rgb2gray(Right_eye);
        E_mouth = rgb2gray(E_mouth);
     subplot(2,2,3);
     imshow(Left_eye);
     subplot(2,2,4);
      imshow(Right_eye);

        % Kmeans clustering
        X = E_mouth(:);
        [nr1 nc1 ] = size(E_mouth);
        clustering_id = kmeans(double(X),2,'emptyaction','drop');
       
        k_out = reshape(clustering_id,nr1,nc1);
        
       
        % Segment
        Ism = zeros(nr1,nc1,3);
         Ism(:,:,3) = 255;
         Ism(:,:,3) = 125;
        Ism(:,:,3) = 255;
       
        bwm = k_out-1;
        bwm3 = cat(3,bwm,bwm,bwm);
        Ism(logical(bwm3)) = E_mouth_3(logical(bwm3));
              
        % Template matching using correlation coefficient
        % Left eye
        % Resize to standard size
        Left_eye =  imresize(Left_eye,[dim(1,1) dim(1,2)]);
        c1 =match_DB(Left_eye,DBL);
        subplot(2,2,3)
        title(cl{c1})
       
       
        % Right eye
        % Resize to standard size
        Right_eye =  imresize(Right_eye,[dim(2,1) dim(2,2)]);
        c2 = match_DB(Right_eye,DBR);
        subplot(2,2,4)
        title(cl{c2})
       
        %% Mouth
        % Resize to standard size
        E_mouth =  imresize(E_mouth,[dim(3,1) dim(3,2)]);
        c3 = match_DB(E_mouth,DBM);
       
       %% Drowsiness detection
        if c1 == 2
            LC = LC+1;
            if c1p == 1
                TC = TC+1;
            end
        end
        if c2==2
            RC = RC+1;
        end
        if c3 == 1
            MC = MC + 1;
        end

        TF = TF + 1;
        toc
        if toc>8
            Feature = [LC/TF RC/TF MC/TF TC]
            species = svmclassify(svmStruct,Feature);
           

            tic
            % Initialise vector
            LC = 0; %
            RC = 0; %
            MC = 0; %
            TF = 0; %
            TC = 0; %
        end
        subplot(2,2,1);
        if strcmpi(species,'Fatigue')
            text(20,20,species,'fontsize',14,'color','r','Fontweight','bold')
            beep;
        else
            text(20,20,species,'fontsize',14,'color','g','Fontweight','bold')
        end
        c1p = c1;
        pause(0.00005)
    end
end

