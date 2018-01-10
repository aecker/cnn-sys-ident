import textwrap


class Messager:
    def msg(self, *msg, depth=0, **kwargs):
        lines = []
        for line in ' '.join(map(str, msg)).split('\n'):
            lines.extend(textwrap.wrap(line))
        tmp = self.__class__.__name__.ljust(25)
        if len(tmp) > 25:
            tmp = tmp[:22] + '...'
        msg = tmp + ('  | ' + depth * '\t') \
                    + ('\n'.ljust(25) + '   | ' + depth * '\t').join(lines)
        print(msg, **kwargs)
