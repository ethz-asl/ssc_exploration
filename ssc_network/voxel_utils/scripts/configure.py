from distutils import sysconfig
print(sysconfig.get_config_var('LDSHARED').replace("gcc", "g++"))
