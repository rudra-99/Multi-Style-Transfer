from libraries import *
class singleton_variable(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(singleton_variable, cls).__new__(cls)
        return cls.instance
class singleton:
    def __init__(self,x) -> None:
        self.singleton_var = singleton_variable()
        self.singleton_var.var = tf.Variable(x)
    def get_instance(self):
        return self.singleton_var.var
        
       
    
    
