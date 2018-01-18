from inspect import isclass
import datajoint as dj

from ..utils.logging import Messager
from ..utils.data import key_hash
from cnn_sys_ident import architectures


class Config(Messager):
    """Abstract base class for configurations.
    
    This class allows mapping configurations of components of a common type
    but different parameters onto a common database layout, such that
    downstream tables/analyses can be done transparently without being
    specific to a certain type.
    
    For example, this layout allows us to analyze models with different cores
    (e.g. CNN, RNN), which have very different parameters in a common
    framework.
    
    Each config is identified by a unique hash and has a type. The type
    defines which subtable contains the parameters and which class is used
    to instantiate the component. For an example, see database/models.py.
    """
    _type = None

    @property
    def definition(self):
        return """
        # parameters for {cn}

        {ct}_hash                   : varchar(256) # unique identifier for configuration 
        ---
        {ct}_type                   : varchar(50)  # type
        {ct}_ts=CURRENT_TIMESTAMP   : timestamp    # automatic
        """.format(ct=self._type, cn=self.__class__.__name__)

    @property
    def parts(self):
        for member in dir(self):
            if isclass(getattr(self, member)) and issubclass(getattr(self, member), dj.Part):
                yield getattr(self, member)

    def fill(self):
        type_name = self._type + '_type'
        hash_name = self._type + '_hash'
        for rel in self.parts:
            self.msg('Checking', rel.__name__)
            for key in rel().content:
                key[type_name] = rel.__name__
                key[hash_name] = key_hash(key)

                if not key in rel().proj():
                    self.insert1(key, ignore_extra_fields=True)
                    self.msg('Inserting', key, flush=True, depth=1)
                    rel().insert1(key, ignore_extra_fields=True)

    def instance_type(self, key):
        type_name = self._type + '_type'
        return (self & key).fetch1(type_name)

    def instance(self, key):
        table = getattr(self, self.instance_type(key))
        return table() & key

