from EnhancementThread import EnhancementThread

z_slice=94
z_res=0.43
z_resize=int((z_slice*z_res/0.09)/2)
print(z_resize)
embryo_name='240410ujls113ceh36unc30p1'
output_path='/new_HDD_1/zhaocunmin/NucApp-develop/output/'

para={'raw_project_dir':'/new_HDD_1/zhaocunmin/NucApp-develop/data/',
    'embryo_name':embryo_name,
    'x_raw':512,
    'y_raw':712,
    'z_raw':z_slice,
    'x_resize':256,
    'y_resize':356,
    'z_resize':z_resize,
    'save_project_dir':output_path, 
    'tem_3d_middle_folder':None}
print(para)

enhancement_thread = EnhancementThread(para)

# Run the processing
enhancement_thread.run()