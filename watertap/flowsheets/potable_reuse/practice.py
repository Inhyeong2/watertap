from idaes.models.unit_models.separator import SeparatorData

# Print the available CONFIG keys and their default values
for key in SeparatorData.CONFIG:
    value = SeparatorData.CONFIG[key]
    if hasattr(value, 'default'):
        print(f"{key}: {value.default}")
    else:
        print(f"{key}: {value}")