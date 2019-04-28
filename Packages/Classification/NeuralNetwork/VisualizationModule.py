import os
from os.path import join

class VisualizationModule(object):
    def __init__(self,path,format_file_tupe=None):
        self.dic_files = {}
        list_file = []
        for root,directories,filename in os.walk(path):
            for ifile in filename:
                if format_file_tupe == None:
                    list_file.append(os.path.join(root,ifile))
                elif ifile.endswith(format_file_tupe):
                #elif ifile.split('.')[1] == format_file:
                    list_file.append(os.path.join(root,ifile))
            self.dic_files[root] = list_file 
            
#            if not bool(directories):
#                for ifile in filename:
#                    name_file , type_file = ifile.split('.')
#                    self.dic_files[name_file] = [join(root,ifile),type_file]
#            else:    
#                for idirec in directories:
#                    for ifile in filename:
#                        name_file , type_file = ifile.split('.')
#                        if format_file == type_file:
#                            self.dic_files[name_file] = join(root,ifile)
#                        elif format_file == None:
#                            self.dic_files[name_file] = [join(root,ifile),type_file]
#                    self.dic_direc[idirec] = self.dic_files
#            self.dic_root[root] = self.dic_direc
        
        if not bool(self.dic_files):
            print 'CAUTION: There arent files in ' + str(path)
            
    
    def getFilesPath(self):
        return self.dic_root
        
        
        
                    
                    
                    