import array_to_latex as a2l


class LatexConverter:
    @classmethod
    def convert_to_latex(cls, matrix):
        return a2l.to_ltx(matrix, frmt="{:n}", print_out=False)
