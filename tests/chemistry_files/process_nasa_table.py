count = 0
print("Reading NASA data and reformatting for PHiLiP...")

files = ["XX"]

for filename in files:
    with open(filename+"_data.txt") as fp:
        l = fp.readline()
        char_index = 0
        species_name = ""
        while True:
            if l[char_index].isspace():
                break
            else:
                species_name+=l[char_index]
            char_index+=1
            
        l = fp.readline()
        char_index = 0
        space_count = 0
        species_weight = ""
        while True:
            if space_count == 9:
                while True:
                    if l[char_index].isspace():
                        break
                    species_weight += l[char_index]
                    char_index += 1
                break
                    
            if l[char_index].isspace():
                space_count += 1
            char_index += 1
        
        all_coeffs = []
        all_temps = []
        for lines in range(3):
            l = fp.readline()
            char_index = 0
            temp = ""
            while l[char_index]!=' ':
                temp += l[char_index]
                char_index += 1
            if len(all_temps) > 0:
                if temp != all_temps[-1]:
                    all_temps.append(temp)
            else:
                all_temps.append(temp)
            temp = ""
            char_index+=1
            while l[char_index]!='7':
                temp += l[char_index]
                char_index += 1
            if len(all_temps) > 0:
                if temp != all_temps[-1]:
                    all_temps.append(temp)
            else:
                all_temps.append(temp)
            temp = ""
            
            enthalpy_offset = ''
            space_count = 0
            while True:
                if space_count == 9:
                    while True:
                        if l[char_index]=='\n':
                            break
                        enthalpy_offset += l[char_index]
                        char_index += 1
                    break
                        
                if l[char_index].isspace():
                    space_count += 1
                char_index += 1
            coeffs = []
            for i in range(2):
                l = fp.readline()
                char_index = 0
                coeff = ''
                while True:
                    if l[char_index] == '-' and len(coeff) > 0:
                        if coeff[-1]!='e':
                            coeffs.append(coeff)
                            coeff ='-'
                            char_index += 1
                        else:
                            coeff += l[char_index]
                            char_index += 1
                    elif l[char_index]==' ':
                        coeffs.append(coeff)
                        coeff =''
                        char_index += 1
                    elif l[char_index]=='\n':
                        coeffs.append(coeff)
                        coeff =''
                        break
                    elif l[char_index]=='D':
                        coeff += 'e'
                        char_index += 1
                    else:
                        coeff += l[char_index]
                        char_index += 1
            all_coeffs.append(coeffs)

    with open(filename+".kinetics","w") as fp:
        print("\nWriting out " + filename + " data...")
        fp.write(species_name+"\n")
        fp.write(species_weight+"\n")
        fp.write(enthalpy_offset+"\n")
        for temp in all_temps:
            fp.write(temp+" ")
        fp.write("\n")
        for a in range(len(all_coeffs)):
            for b in range(len(all_coeffs[a])):
                fp.write(all_coeffs[a][b])
                if b != len(all_coeffs[a]) - 1:
                    fp.write(" ")
            fp.write("\n")
        