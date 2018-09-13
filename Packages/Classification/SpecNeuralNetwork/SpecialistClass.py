from OneSpecialistClass import OneSpecialistClass as SC

class SpecialistClass():
    """docstring for SpecialistClass."""
    def __init__(self,resultPath,trnParams,num_specialist):
        #super(SpecialistClass, self).__init__()
        #self.arg = arg
        self.specialist = {}
        for ispec in range(num_specialist):
            self.specialist[ispec] = SC(resultPath,trnParams,ispec)


    def prepare_target(self,trgt,spec_num):
        index_spec, =np.where(all_trgt==spec_num)
        spec_trgt = trgt*0
        spec_trgt[index_spec] = 1
        return spec_trgt

    def train_specialist_fold(self,data,trgt,fold,spec_num):
        spec_trgt = self.prepare_target(trgt,ispec)
        model = self.speacialist[spec_num].train(data,spec_trgt,fold)
        return model

    def train_specialist(self,data,trgt,spec_num):
        spec_trgt = self.prepare_target(trgt,ispec)
        model = self.speacialist[spec_num].train_n_folds(data,spec_trgt)
        return model
