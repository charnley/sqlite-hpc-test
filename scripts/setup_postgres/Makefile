
prefix=$$HOME/opt/postgres
data=$$HOME/pg/data
repo=postgres.git

postgres.git:
	git clone --depth 1 https://github.com/postgres/postgres.git postgres.git

configure:
	PYTHON=$$(which python) && cd postgres.git && ./configure --prefix $$HOME/opt/postgres --with-python PYTHON=$$PYTHON

build:
	cd postgres.git && make -j 4

test:
	cd ${repo} && make check


install:
	cd postgres.git && make install

create_database:
	${prefix}/bin/initdb \ 
		-D ${data}

start:
	${prefix}/bin/postgres -D ${data}

# Avoid global libs
# ./configure --disable-rpath [other options]
# export LD_RUN_PATH='$ORIGIN/../lib'
# make
# make install


# https://stackoverflow.com/questions/38466190/cant-connect-to-postgresql-on-port-5432

# You have to edit pg_hba.conf to accept all requests within the network
# #TYPE  DATABASE        USER            ADDRESS                 METHOD 
# host    all             all             0.0.0.0/0                trust
# host    all             all             ::1/128                  trust

# postgresql.conf
# listen_addresses = '*'



