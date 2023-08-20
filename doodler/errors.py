#!/usr/bin/env python3
# Copyright (c) 2023, Joseph M. Rutherford

import copy

class Recoverable(Exception):
    '''Possibly recoverable error conditions derive from this class'''

    def __init__(self,error):
        '''Retains a copy of the passed argument'''
        self.content = None
        if type(error) is str: # Simple message argument
            self.content = error # strings are immutable, so the message is fixed
        elif isinstance(error,Exception):
            self.content = copy.deepcopy(error)
        else:
            self.content = '\n'.join(['Unknown condition:',repr(error)])
        super().__init__(self,'\n'.join(['Possibly recoverable condition:',str(self.content)]))

class Unrecoverable(Exception):
    '''Definitely unrecoverable error conditions derive from this class'''

    def __init__(self,error):
        '''Retains a copy of the passed argument'''
        self.content = None
        if type(error) is str: # simple message argument
            self.content = error
        elif isinstance(error,Recoverable): # error re-interpreted as Unrecoverable
            self.content = copy.deepcopy(error.content)
        elif isinstance(error,Exception): # any other exception type
            self.content = copy.deepcopy(error)
        else:
            self.content = '\n'#.join(['Unknown error:',repr(error)])
        super().__init__(self,'\n'.join(['Unrecoverable error:',str(self.content)]))

class NeverImplement(Unrecoverable):
    '''Raise when a method is called that should never be implemented, such as abstract parent class function'''

    def __init__(self,error):
        '''Parent class retains the error message string'''
        if type(error) is str:
            super.__init__(self,error)
        else:
            raise Unrecoverable('NeverImplement requires a string argument')