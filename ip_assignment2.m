% Begin initialization code - DO NOT EDIT
function varargout = ip_assignment2(varargin)
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @ip_assignment2_OpeningFcn, ...
                   'gui_OutputFcn',  @ip_assignment2_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before ip_assignment is made visible.
function ip_assignment2_OpeningFcn(hObject, eventdata, handles, varargin)
% Choose default command line output for ip_assignment
handles.output = hObject;
% Update handles structure
guidata(hObject, handles);


% --- Outputs from this function are returned to the command line.
function varargout = ip_assignment2_OutputFcn(hObject, eventdata, handles) 
% Get default command line output from handles structure
varargout{1} = handles.output;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------------------------LOAD-----------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function load_Callback(hObject, eventdata, handles)
%defining the variables as global to use them across functions
%current_image is the axes1 being displayed on the axes
global current_image backup_image 

current_image=imread('C:\Users\ADITYA MITTAL\Desktop\Images\Blurry3_1.jpg'); % load image
backup_image=current_image; % for backup process for undo all option
cla; %remove the previous image
axes(handles.axes1); % handle to axes
imshow(current_image); %display image



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------------------------UNDO-----------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function undo_Callback(hObject, eventdata, handles)
%defining the variables as global to use them across functions
%current_image is the axes1 being displayed on the axes
global current_image prev_image

%switching current_image and prev_image
temp = current_image; 
current_image = prev_image;
prev_image = temp;

cla; %remove the previous image
axes(handles.axes1); % handle to axes
imshow(current_image); %display image


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%----------------------------UNDO ALL-----------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function undo_all_Callback(hObject, eventdata, handles)
%defining the variables as global to use them across functions
%current_image is the axes1 being displayed on the axes
global current_image prev_image backup_image
prev_image = current_image; %as current_image is going to be changed%backup for Undo option
current_image = backup_image; %backup_image is created while loading of image for undo_all option
cla; %remove the previous image
axes(handles.axes1); % handle to axes
imshow(current_image); %display image

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------SAVE THE AXES1--------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Executes on button press in save.
function save_Callback(hObject, eventdata, handles)
%defining the variables as global to use them across functions
%current_image is the axes1 being displayed on the axes
global current_image
[file,path] = uiputfile('output_im.png','Save the output');%asking path and filename from user
filename = strcat(path,file);
imwrite(current_image,filename);%saving the file


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------------------Inverse BLUR--------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Executes on button press in inverse.
function inverse_Callback(hObject, eventdata, handles)
global current_image prev_image
prev_image = current_image; %backup image to be used for undo change

%Computing the FFT of the axes1 centered at middle
FFT_curr_image = fft2(im2double(current_image));

%Importing a blur filter centered at middle
blur_kernel_1_unpadded=im2double(imread('C:\Users\ADITYA MITTAL\Desktop\Blur_kernels\Cho_Deblur\Blurr1_1_result_Cho_5.png.psf.png')); % load blur kernel
norm_factor = sum(sum(blur_kernel_1_unpadded));
blur_kernel_1_unpadded_normalised = blur_kernel_1_unpadded(:,:,1)/norm_factor(:,:,1);
%Padding blur kernel with zeros to make it same size as current axes1
blur_kernel_1 = zeros(size(current_image,1),size(current_image,2));
for i=1:size(blur_kernel_1_unpadded_normalised,1)
    for j=1:size(blur_kernel_1_unpadded_normalised,2)
        blur_kernel_1(i,j) = blur_kernel_1_unpadded_normalised(i,j);
    end
end
FFT_blur_kernel_1 = fft2(blur_kernel_1); %Computing FFT of blur kernel

%Fourier of restored axes1 recieved by divding by degradation function
%Limiting the radius in which H is to be considered
F_cap = zeros(size(FFT_curr_image,1),size(FFT_curr_image,2),3); %Initialising an image F_cap which will be the Fourier of restored image
radius = get(handles.inverse_slide, 'Value')*((size(FFT_curr_image,1)/2)^2+(size(FFT_curr_image,2)/2)^2)^0.5
%get input which will be the radius in which H is to be considered
for i=1:size(FFT_blur_kernel_1,1)
    for j=1:size(FFT_blur_kernel_1,2)
        % we check if the (i,j) is in the specified radius
        if i < size(FFT_curr_image,1)/2
            k = i;
        else
            k = size(FFT_curr_image,1) - i;
        end
        if j < size(FFT_curr_image,2)/2
            l = j;
        else
            l = size(FFT_curr_image,2) - j;
        end
        if k^2+l^2 < radius^2
            F_cap(i,j,1) = FFT_curr_image(i,j,1)/FFT_blur_kernel_1(i,j);
            F_cap(i,j,2) = FFT_curr_image(i,j,2)/FFT_blur_kernel_1(i,j);
            F_cap(i,j,3) = FFT_curr_image(i,j,3)/FFT_blur_kernel_1(i,j);
        else
            F_cap(i,j,1) = FFT_curr_image(i,j,1);
            F_cap(i,j,2) = FFT_curr_image(i,j,2);
            F_cap(i,j,3) = FFT_curr_image(i,j,3);
        end
    end
end

%The inverese fourier transform of the F_cap is calculated which will be
%our output deblurred axes1
current_image = real(ifft2(F_cap));

InputImage = im2double(imread('C:\Users\ADITYA MITTAL\Desktop\Images\GroundTruth3_1_1.jpg'));
MSE = sum(sum(sum((InputImage-current_image).^2)))/(size(current_image,1)*size(current_image,2)*size(current_image,3));
PSNR = 10*log10(256*256/MSE)
L = 255;
c1 = (0.01*L).^2;%c1 and c2 selected randomly to stabilize the division with weak denominator
c2 = (0.03*L).^2;
mu_current_image = mean(mean(mean(current_image)));
mu_InputImage = mean(mean(mean(InputImage)));
var_current_image = mean(mean(mean((current_image-mu_current_image).^2)));
var_InputImage = mean(mean(mean((InputImage-mu_InputImage).^2)));
cov_current_image_InputImage = mean(mean(mean((current_image-mu_current_image).*(InputImage-mu_InputImage))));
SSIM = ((2*mu_current_image*mu_InputImage + c1)*(2*cov_current_image_InputImage + c2))/((mu_current_image + mu_InputImage + c1)*(var_current_image + var_InputImage +c2))

cla; %remove the previous image
axes(handles.axes1); % handle to axes
imshow(current_image); %display image


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------------------Weiner Filter--------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Executes on button press in weiner.
function weiner_Callback(hObject, eventdata, handles)
global current_image prev_image
prev_image = current_image; %backup image to be used for undo change

%Computing the FFT of the axes1 centered at middle
FFT_curr_image = fft2(im2double(current_image));

%Importing a blur filter centered at middle
blur_kernel_1_unpadded=im2double(imread('C:\Users\ADITYA MITTAL\Desktop\Blur_kernels\Cho_Deblur\Blurr1_1_result_Cho_5.png.psf.png')); % load blur kernel
norm_factor = sum(sum(blur_kernel_1_unpadded));%normalisation factor
blur_kernel_1_unpadded_normalised = blur_kernel_1_unpadded(:,:,1)/norm_factor(:,:,1);%normalisation
%Padding blur kernel with zeros to make it same size as current axes1
blur_kernel_1 = zeros(size(current_image,1),size(current_image,2));
for i=1:size(blur_kernel_1_unpadded_normalised,1)
    for j=1:size(blur_kernel_1_unpadded_normalised,2)
        blur_kernel_1(i,j) = blur_kernel_1_unpadded_normalised(i,j);
    end
end
FFT_blur_kernel_1 = fft2(blur_kernel_1); %Computing FFT of blur kernel

%Fourier of restored axes1 recieved
K = get(handles.weiner_slide, 'Value')*0.5%get value of K from slider
weiener_expression = conj(FFT_blur_kernel_1)./((abs(FFT_blur_kernel_1)).^2+K);
F_cap(:,:,1) = FFT_curr_image(:,:,1).*weiener_expression;
F_cap(:,:,2) = FFT_curr_image(:,:,2).*weiener_expression;
F_cap(:,:,3) = FFT_curr_image(:,:,3).*weiener_expression;
        

%The inverese fourier transform of the F_cap is calculated which will be
%our output deblurred axes1
current_image = real(ifft2(F_cap));

%Calculating PSNR and SSIM
InputImage = im2double(imread('C:\Users\ADITYA MITTAL\Desktop\Images\GroundTruth3_1_1.jpg'));
MSE = sum(sum(sum((InputImage-current_image).^2)))/(size(current_image,1)*size(current_image,2)*size(current_image,3));
PSNR = 10*log10(256*256/MSE)
L = 255;
c1 = (0.01*L).^2;%c1 and c2 selected randomly to stabilize the division with weak denominator
c2 = (0.03*L).^2;
mu_current_image = mean(mean(mean(current_image)));%mean of image 1
mu_InputImage = mean(mean(mean(InputImage)));%mean of image 2
var_current_image = mean(mean(mean((current_image-mu_current_image).^2)));%var of image 1
var_InputImage = mean(mean(mean((InputImage-mu_InputImage).^2)));%var of image 2
cov_current_image_InputImage = mean(mean(mean((current_image-mu_current_image).*(InputImage-mu_InputImage))));
%co variance of both images
SSIM = ((2*mu_current_image*mu_InputImage + c1)*(2*cov_current_image_InputImage + c2))/((mu_current_image + mu_InputImage + c1)*(var_current_image + var_InputImage +c2))


cla; %remove the previous image
axes(handles.axes1); % handle to axes
imshow(current_image); %display image


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-----------------Constrained Low Pass Filter------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Executes on button press in clsf.
function clsf_Callback(hObject, eventdata, handles)
global current_image prev_image
prev_image = current_image; %backup image to be used for undo change

%Computing the FFT of the axes1 centered at middle
FFT_curr_image = fft2(im2double(current_image));

%Importing a blur filter centered at middle
blur_kernel_1_unpadded=im2double(imread('C:\Users\ADITYA MITTAL\Desktop\Blur_kernels\Cho_Deblur\Blurr1_1_result_Cho_5.png.psf.png')); % load blur kernel
norm_factor = sum(sum(blur_kernel_1_unpadded));%normalisation factor
blur_kernel_1_unpadded_normalised = blur_kernel_1_unpadded(:,:,1)/norm_factor(:,:,1);%normalisation
%Padding blur kernel with zeros to make it same size as current axes1
blur_kernel_1 = zeros(size(current_image,1),size(current_image,2));
for i=1:size(blur_kernel_1_unpadded_normalised,1)
    for j=1:size(blur_kernel_1_unpadded_normalised,2)
        blur_kernel_1(i,j) = blur_kernel_1_unpadded_normalised(i,j);
    end
end
FFT_blur_kernel_1 = fft2(blur_kernel_1); %Computing FFT of blur kernel

%Laplacian matrix
laplace = [0 -1 0; -1 4 -1; 0 -1 0];
laplace_padded = zeros(size(current_image,1),size(current_image,2));%initialise padded laplacian array
laplace_padded(1:3,1:3) = laplace;
laplace_fft = fft2(laplace_padded); %computing DFT of the laplace matrix
laplace_fft_sqd = (abs(laplace_fft)).^2;

%Fourier of restored axes1 recieved
gamma = get(handles.weiner_slide, 'Value')*0.001%get value of gamma from slider
clsf_expression = conj(FFT_blur_kernel_1)./((abs(FFT_blur_kernel_1)).^2+gamma*laplace_fft_sqd);
F_cap(:,:,1) = FFT_curr_image(:,:,1).*clsf_expression;
F_cap(:,:,2) = FFT_curr_image(:,:,2).*clsf_expression;
F_cap(:,:,3) = FFT_curr_image(:,:,3).*clsf_expression;
        

%The inverese fourier transform of the F_cap is calculated which will be
%our output deblurred axes1
current_image = real(ifft2(F_cap));

%Calculating PSNR and SSIM
InputImage = im2double(imread('C:\Users\ADITYA MITTAL\Desktop\Images\GroundTruth3_1_1.jpg'));
MSE = sum(sum(sum((InputImage-current_image).^2)))/(size(current_image,1)*size(current_image,2)*size(current_image,3));
PSNR = 10*log10(256*256/MSE)
L = 255;
c1 = (0.01*L).^2;%c1 and c2 selected randomly to stabilize the division with weak denominator
c2 = (0.03*L).^2;
mu_current_image = mean(mean(mean(current_image)));%mean of image 1
mu_InputImage = mean(mean(mean(InputImage)));%mean of image 2
var_current_image = mean(mean(mean((current_image-mu_current_image).^2)));%var of image 1
var_InputImage = mean(mean(mean((InputImage-mu_InputImage).^2)));%var of image 2
cov_current_image_InputImage = mean(mean(mean((current_image-mu_current_image).*(InputImage-mu_InputImage))));
%co variance of both images
SSIM = ((2*mu_current_image*mu_InputImage + c1)*(2*cov_current_image_InputImage + c2))/((mu_current_image + mu_InputImage + c1)*(var_current_image + var_InputImage +c2))


cla; %remove the previous image
axes(handles.axes1); % handle to axes
imshow(current_image); %display image

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------------------SLIDES------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Executes on slider movement.
function weiner_slide_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function weiner_slide_CreateFcn(hObject, eventdata, handles)
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

% --- Executes on slider movement.
function clsf_slide_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function clsf_slide_CreateFcn(hObject, eventdata, handles)
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

% --- Executes on slider movement.
function inverse_slide_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function inverse_slide_CreateFcn(hObject, eventdata, handles)
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------------FUNCTION TO CALCULATE DFT-----------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [output]=find_DFT(image)
M=size(image,1);%size of image in x dimension
N=size(image,2);%size of image in y dimension
if size(size(image),2) == 3
    output = zeros(size(image,1),size(image,2),size(image,3),'double');%initilise output same size as image
    for u=1:M
        for v=1:N
            %calculating DFT for (u,v)
            for x=1:M
                for y=1:N
                    output(u,v,:)=output(u,v,:)+double(image(x,y,:)).*exp((-1i).*2.*pi.*((x-1).*(u-1)./M)+(y-1).*(v-1)./N);
                end
            end
            output(u,v,:)
        end 
    end
else
    output = zeros(size(image,1),size(image,2));%initilise output same size as image
    for u=1:M
        for v=1:N
            %calculating DFT for (u,v)
            for x=1:M
                for y=1:N
                    output(u,v)=output(u,v)+image(x,y).*exp((-1i).*2.*pi.*((x-1).*(u-1)./M)+(y-1).*(v-1)./N);
                end
            end
        end 
    end
end

function [output]=find_IDFT(dft)
M=size(dft,1);%size of image in x dimension
N=size(dft,2);%size of image in y dimension

ndim2 = size(size(dft),2)
if size(size(dft),2) == 3
    output = zeros(size(dft,1),size(dft,2),size(dft,3));%initilise output same size as image

    for u=1:M
        for v=1:N
            %calculating DFT for (u,v)
            for x=1:M
                for y=1:N
                    output(u,v,:)=output(u,v,:)+(dft(x,y,:).*exp((1j).*2.*pi.*((x-1).*(u-1)./M)+(y-1).*(v-1)./N))./M./N;
                end
            end
        end 
    end
else 
    output = zeros(size(dft,1),size(dft,2));%initilise output same size as image

    for u=1:M
        for v=1:N
            %calculating DFT for (u,v)
            for x=1:M
                for y=1:N
                    output(u,v)=output(u,v)+(dft(x,y).*exp((1j).*2.*pi.*((x-1).*(u-1)./M)+(y-1).*(v-1)./N))./M./N;
                end
            end
        end 
    end
end
