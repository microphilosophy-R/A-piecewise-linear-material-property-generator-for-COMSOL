import CoolProp as CP
import matplotlib
import pandas as pd
import numpy as np
import os
import openpyxl
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


class PiecewisePropertyGenerator():
    def __init__(self, species='hydrogen', backend='REFPROP',
                 T_min=20, T_max=300, Pref=1e6, resolution=0.2, tolerance=0.01,
                 property_func='cpmolar', output_dir=None):
        self.species = species
        self.backend = backend
        self.T_min = T_min
        self.T_max = T_max
        self.Pref = Pref
        self.resolution = resolution
        self.tolerance = 0.01
        self.property_func = property_func
        self.output_dir = output_dir or os.getcwd()

        # 初始化 CoolProp state
        try:
            self.state = CP.AbstractState(backend, species)
            print(f"Initialized AbstractState with {backend}, {species}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize CoolProp AbstractState: {e}")

        # 物性函数映射
        self._prop_method = self._get_property_method(property_func)
        # 温度网格
        n_points = int((T_max - T_min) / resolution)
        if n_points < 4:
            raise ValueError(f"Too few points ({n_points}) to construct property curve. "
                             f"Please reduce 'resolution' or increase temperature range.")
        self.T = np.linspace(T_min, T_max, n_points)

        # 获取物性数据
        try:
            self.tp = self._fetch_property()
        except Exception as e:
            raise RuntimeError(f"Failed to fetch property data: {e}")

    def _get_property_method(self, prop_name):
        """根据属性名称获取对应的 CoolProp 方法"""
        method_map = {
            'cpmolar': self.state.cpmolar,
            'cvmolar': self.state.cvmolar,
            'density': self.state.rhomass,
            'enthalpy': self.state.hmolar,
            'entropy': self.state.smolar,
            'conductivity': self.state.conductivity,
            'viscosity': self.state.viscosity
        }

        if prop_name in method_map:
            return method_map[prop_name]
        else:
            raise ValueError(f"Unsupported property function: {prop_name}")

    def _get_unit(self):
        """根据物性返回对应的单位"""
        units = {
            'cpmolar': 'J/mol/K',
            'cvmolar': 'J/mol/K',
            'density': 'kg/m^3',
            'enthalpy': 'J/mol',
            'entropy': 'J/mol/K',
            'conductivity': 'W/m/K',
            'viscosity': 'Pa·s'
        }
        return units.get(self.property_func, 'Unknown Unit')

    def _fetch_property(self):
        """直接调用绑定的物性方法"""
        tp = []
        for T in self.T:
            try:
                # 更新状态
                self.state.update(CP.PT_INPUTS, self.Pref, T)

                # 调用绑定的方法（注意括号）
                value = self._prop_method()

                # 添加数据
                tp.append(value)

            except Exception as e:
                print(f"Error at T={T:.2f}K: {e}")
                tp.append(np.nan)  # 或者抛出异常、跳过该温度点

        if not tp:
            print("current _prop_method", self._prop_method)
            raise RuntimeError("No data fetched in _fetch_property(). Check CoolProp state or property method.")

        return np.array(tp)

    def adaptive_segmentation(self, segments, tolerance=None, max_depth=10, depth=0):
        """自适应分段算法，使用相对容差"""
        if tolerance is None:
            # 默认使用 tp 最大值的 1%
            tolerance = np.max(np.abs(self.tp)) * 0.001

        refined_segments = []
        for t_low, t_high in segments:
            mask = (self.T >= t_low) & (self.T <= t_high)
            T_seg = self.T[mask]
            tp_seg = np.array(self.tp)[mask]

            if len(T_seg) < 4 or depth >= max_depth:
                refined_segments.append((t_low, t_high))
                continue

            coeff = np.polyfit(T_seg, tp_seg, deg=3)
            poly = np.poly1d(coeff)
            error = np.max(np.abs(poly(T_seg) - tp_seg))

            if error > tolerance:
                mid = (t_low + t_high) / 2
                refined_segments.extend(
                    self.adaptive_segmentation([(t_low, mid), (mid, t_high)], tolerance, max_depth, depth + 1)
                )
            else:
                refined_segments.append((t_low, t_high))
        return refined_segments

    def find_segments(self):
        """自动寻找断点并生成初始分段"""
        # 寻找极值点
        peaks_max, _ = find_peaks(self.tp, prominence=0.1)
        peaks_min, _ = find_peaks([-y for y in self.tp], prominence=0.1)

        breakpoints = np.sort(np.concatenate((peaks_max, peaks_min)))
        T_breakpoints = self.T[breakpoints]

        # 添加起点终点
        T_breakpoints = np.insert(T_breakpoints, 0, self.T_min)
        T_breakpoints = np.append(T_breakpoints, self.T_max)

        # 初始分段
        segments = [(T_breakpoints[i], T_breakpoints[i + 1]) for i in range(len(T_breakpoints) - 1)]
        return self.adaptive_segmentation(segments, tolerance=None)

    def fit_coefficients(self):
        """拟合每段的多项式系数"""
        coefficients = []
        segments = self.find_segments()

        for t_low, t_high in segments:
            mask = (self.T >= t_low) & (self.T <= t_high)
            T_segment = self.T[mask]
            tp_segment = np.array(self.tp)[mask]

            if len(T_segment) < 4:
                print(f"Segment [{t_low}, {t_high}] too small for cubic fit")
                continue

            coeff = np.polyfit(T_segment, tp_segment, deg=3)
            coefficients.append(coeff)

        return coefficients, segments

    def save_to_file(self, coefficients, segments):
        """保存结果到 Excel 和 txt 文件"""
        df_coeff = pd.DataFrame(coefficients, columns=['a3', 'a2', 'a1', 'a0'])
        df_coeff['T_low'] = [seg[0] for seg in segments]
        df_coeff['T_high'] = [seg[1] for seg in segments]
        df_coeff = df_coeff[['T_low', 'a0', 'a1', 'a2', 'a3', 'T_high']]

        filename = f"{self.species}_{self.property_func}_{self.Pref / 1e6}MPa_{self.T_min}-{self.T_max}K"
        filename = filename.replace('.', '_')

        # 输出 Excel
        df_coeff.to_excel(os.path.join(self.output_dir, filename + ".xlsx"), index=False, float_format='%.6e')
        print(f"Saved to {filename}.xlsx")

        # 输出 TXT
        df_coeff.to_csv(os.path.join(self.output_dir, filename + ".txt"), sep=' ', index=False,
                        float_format='%.6e', header=False)
        print(f"Saved to {filename}.txt")

    def save_polynomial_expressions(self, coefficients, segments):
        """生成并保存多项式表达式到文件"""

        def generate_expression(coeffs):
            a3, a2, a1, a0 = coeffs
            return f"{a3:.6e}*x^3 + {a2:.6e}*x^2 + {a1:.6e}*x + {a0:.6e}"

        expressions = [generate_expression(coeff) for coeff in coefficients]

        df_poly = pd.DataFrame({
            'T_low': [seg[0] for seg in segments],
            'T_high': [seg[1] for seg in segments],
            'Polynomial': expressions
        })

        filename = f"{self.species}_{self.property_func}_{self.Pref / 1e6}MPa_{self.T_min}-{self.T_max}K"
        filename = filename.replace('.', '_')

        # 输出 Excel
        df_poly.to_excel(os.path.join(self.output_dir, filename + "_equations.xlsx"), index=False)
        print(f"Saved polynomial equations to {filename}_equations.xlsx")

        # 输出 TXT
        df_poly[['T_low', 'T_high', 'Polynomial']].to_csv(
            os.path.join(self.output_dir, filename + "_equations.txt"),
            sep=' ', index=False, header=False
        )
        print(f"Saved polynomial equations to {filename}_equations.txt")

    def plot_results(self, coefficients, segments, save_path=None):
        """绘制原始数据与拟合曲线，并可选保存图像"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.T, self.tp, label='Original', color='blue', linestyle='--')

        for i, (t_low, t_high) in enumerate(segments):
            T_segment = np.linspace(t_low, t_high, 100)
            poly = np.poly1d(coefficients[i])
            plt.plot(T_segment, poly(T_segment), label=f'Fit Segment {i + 1}')

        plt.xlabel('Temperature (K)')
        plt.ylabel(f'{self.property_func} ({self._get_unit()})')
        plt.title(f'Piecewise Polynomial Fit of {self.property_func}')
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def run(self):
        """运行完整流程"""
        coefficients, segments = self.fit_coefficients()
        self.save_to_file(coefficients, segments)
        self.save_polynomial_expressions(coefficients, segments)

        # 构建图像保存路径
        filename_base = f"{self.species}_{self.property_func}_{self.Pref / 1e6}MPa_{self.T_min}-{self.T_max}K"
        filename_base = filename_base.replace('.', '_')
        plot_path = os.path.join(self.output_dir, filename_base + ".png")

        # 绘图并保存
        self.plot_results(coefficients, segments, save_path=plot_path)

    @classmethod
    def generate_all_properties(cls, species='hydrogen', backend='REFPROP',
                                T_min=20, T_max=350, Pref=1e6, resolution=0.2,
                                output_dir=None):
        props = ['cpmolar', 'cvmolar', 'density', 'enthalpy', 'entropy', 'conductivity', 'viscosity']
        for prop in props:
            print(f"Generating data for property: {prop}")
            generator = cls(species=species, backend=backend,
                            T_min=T_min, T_max=T_max,
                            Pref=Pref, resolution=resolution,
                            property_func=prop, output_dir=output_dir)
            coefficients, segments = generator.fit_coefficients()
            generator.save_to_file(coefficients, segments)
            generator.save_polynomial_expressions(coefficients, segments)
            generator.plot_results(coefficients, segments, save_path=os.path.join(output_dir, f"{species}_{prop}.png"))


p = PiecewisePropertyGenerator('hydrogen', 'REFPROP', 20, 350,
                               2e6, 0.2, property_func='enthalpy',
                               output_dir=r"C:\本地文件-COMSOL建模\正仲氢分段热力学性质\thermal_properties(molar)")

# p.run()
# hydrogen, PARAHYD
p.generate_all_properties('hydrogen', 'REFPROP', 20, 350,
                          2e6, 0.2,
                          output_dir=r"C:\本地文件-COMSOL建模\正仲氢分段热力学性质\thermal_properties(molar)")
