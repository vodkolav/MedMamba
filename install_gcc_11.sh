#To install GCC version 11 on an Ubuntu system that already has GCC version 13, you can use the following steps:
#Add the Toolchain PPA: First, you may want to add a PPA (Personal Package Archive) that contains older versions of GCC.
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update

#Next, you can install GCC version 11 specifically.
sudo apt install gcc-11 g++-11

#Set GCC 11 as the default compiler, you can update the alternatives system:
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 60 --slave /usr/bin/g++ g++ /usr/bin/g++-11

#If you have multiple GCC versions installed, you can choose which one to use:
sudo update-alternatives --config gcc
# This command will prompt you to select the version you want to use by entering the selection number.

# Finally, check the installed version of GCC to ensure it's correctly set up.
gcc --version
