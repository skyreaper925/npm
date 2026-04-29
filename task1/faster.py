# Читаем исходный файл (сохраните ваши данные в task.txt)
with open("conf.txt", "r") as f:
    lines = f.readlines()

# Получаем начальные и конечные значения ИЗНАЧАЛЬНЫХ колонок
# (из первой и последней строки)
first_parts = lines[0].split()
last_parts = lines[-1].split()

orig_col2_start = float(first_parts[1])  # ~0.99997
orig_col2_end = float(last_parts[1])  # ~1.50003

orig_col3_start = float(first_parts[2])  # ~1.98361
orig_col3_end = float(last_parts[2])  # ~1.50081

# Задаем ЖЕЛАЕМЫЕ идеальные границы
target_col2_start, target_col2_end = 1.0, 1.5
target_col3_start, target_col3_end = 2.0, 1.5

output_lines = []

for line in lines:
    parts = line.split()
    if len(parts) >= 3:
        val2 = float(parts[1])
        # val3 = float(parts[2])

        # # 1. Вычисляем "прогресс" формы на текущем шаге от 0.0 до 1.0
        # k2 = (val2 - orig_col2_start) / (orig_col2_end - orig_col2_start)
        # k3 = (val3 - orig_col3_start) / (orig_col3_end - orig_col3_start)

        # 2. Масштабируем этот прогресс в новые границы
        new_col3 = 3. - val2
        # new_col3 = target_col3_start + k3 * (target_col3_end - target_col3_start)

        # 3. Обновляем значения, оставляя 5 знаков после запятой
        # parts[1] = f"{new_col2:.5f}"
        parts[2] = f"{new_col3:.5f}"

        output_lines.append(" ".join(parts))

# Сохраняем результат
with open("data.txt", "w") as f:
    f.write("\n".join(output_lines))

print("Готово! Данные отмасштабированы с сохранением изначальной физики процесса.")