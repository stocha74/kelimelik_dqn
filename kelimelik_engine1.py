# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 19:01:01 2024

@author: Cenk Toker
"""

def _normalize_cell_letter(cell):
    if cell is None or cell == "":
        return ""
    return str(cell).upper()


def _is_joker_cell(cell):
    return isinstance(cell, str) and cell != "" and cell.islower()


def _harf_puani_from_cell(cell, harf_puanlari):
    ch = _normalize_cell_letter(cell)
    if ch == "":
        return 0
    if _is_joker_cell(cell):
        return 0
    return harf_puanlari.get(ch, 0)


def _hucre_harf_eslesiyor_mu(cell, harf):
    return _normalize_cell_letter(cell) == str(harf).upper()


def _parse_rack_and_jokers(eldeki_harfler):
    from collections import Counter
    joker_adedi = 0
    rack_counter = Counter()

    if eldeki_harfler is None:
        return rack_counter, joker_adedi

    if isinstance(eldeki_harfler, str):
        for ch in eldeki_harfler:
            if ch in {"?", "*", "_"}:
                joker_adedi += 1
            elif ch.strip():
                rack_counter[ch.upper()] += 1
        return rack_counter, joker_adedi

    for item in eldeki_harfler:
        t = str(item).strip()
        tu = t.upper()
        if tu in {"JOKER", "?", "*", "_"}:
            joker_adedi += 1
        elif t:
            rack_counter[tu] += 1

    return rack_counter, joker_adedi


def _joker_indislerini_bul(stoktan_dus, eldeki_harfler):
    rack_counter, joker_adedi = _parse_rack_and_jokers(eldeki_harfler)
    joker_indisleri = []

    for i, harf in enumerate(stoktan_dus):
        hu = str(harf).upper()
        if rack_counter.get(hu, 0) > 0:
            rack_counter[hu] -= 1
        elif joker_adedi > 0:
            joker_adedi -= 1
            joker_indisleri.append(i)
        else:
            # aday akışında bu noktaya gelinmemeli; güvenli fallback
            pass

    return set(joker_indisleri)


def _kelime_can_be_formed_with_jokers(word, letters_counter, joker_count):
    from collections import Counter
    eksik = 0
    wc = Counter(str(word).upper())
    for ch, need in wc.items():
        have = letters_counter.get(ch, 0)
        if need > have:
            eksik += (need - have)
            if eksik > joker_count:
                return False
    return True


def _normalize_input_word(word):
    """Kullanıcı/dataset kaynaklı kelimeleri temizler (boşlukları atar, büyük harfe çevirir)."""
    if word is None:
        return ""
    return "".join(ch for ch in str(word).upper() if ch.strip())


###  BU KOD İLE , KOYULAN BİR KELİME SONRASI TABLODA AÇIĞA ÇIKAN YENİ KELİMELER VERİLİYOR. ###
###          AYNI ZAMANDA BU KELİMELERİ OLUŞTURAN HARFLERİN X VE Y KOORDİNATLARI DÖNÜLÜYOR ###
###                     BU KOORDİNATLAR HARF VE KELİME PUANLAMADA KULLANILACAKLAR          ###
###KOG GELİŞTİRİLDİ VE TAHTA KATSAYILARINI DA DİKKATE ALAN ELDE EDİLEN NİHAİ PUAN DÖNÜLÜYOR### 
def kelime_kontrol(kelime, x_koord, y_koord, orientation , board_old , harf_puanlari , tahta_puanlari2):
    """    
    ###  BU KOD İLE , KOYULAN BİR KELİME SONRASI TABLODA AÇIĞA ÇIKAN YENİ KELİMELER VERİLİYOR. ###
    ###          AYNI ZAMANDA BU KELİMELERİ OLUŞTURAN HARFLERİN X VE Y KOORDİNATLARI DÖNÜLÜYOR ###
    ###                     BU KOORDİNATLAR HARF VE KELİME PUANLAMADA KULLANILACAKLAR          ###
    ###KOG GELİŞTİRİLDİ VE TAHTA KATSAYILARINI DA DİKKATE ALAN ELDE EDİLEN NİHAİ PUAN DÖNÜLÜYOR### 
    """

    
    x_koord1=x_koord
    y_koord1=y_koord

    if orientation=="h" or orientation=="H":
    
        i = 0  # Track position in `kelime`
        olusan_kelimeler=[]
        yler=[]
        xler=[]
        harfler=[]
        puantoplam=0
        puanlardizisi=[]
        for a in kelime:
            y_basl = y_koord
        
            #print("Current letter:", a)
        
            # Look up for the start of the word segment
            y = y_koord - 1
            while y >= 0 and board_old[y][x_koord] != "":
                y -= 1
            y_basl = y + 1
            #print("y_basl=" + str(y_basl))
        
            # Look down for the end of the word segment
            y = y_koord + 1
            while y < len(board_old) and board_old[y][x_koord] != "":
                y += 1
            y_bitis = y - 1
            #print("y_bitiş=" + str(y_bitis))
        
            # Construct the word with letters from `kelime` and existing board letters
            tam_kelime = []
            puan=0
            #puan=[]
            while y_basl <= y_bitis:
                # Insert from `kelime` if this position corresponds to one of its letters
                if y_basl == y_koord and i < len(kelime):
                    yler.append(y_basl)
                    xler.append(x_koord)
                    harfler.append(kelime[i])
                    tam_kelime.append(kelime[i])
                    puan=puan+harf_puanlari[kelime[i]]
                    #puan.append(tahta_puanlari2[y_basl,x_koord])
                    i += 1  # Move to the next letter in `kelime`
                else:
                    # Otherwise, use the letter from `board_old` if available
                    tam_kelime.append(board_old[y_basl][x_koord])
                    yler.append(y_basl)
                    xler.append(x_koord)
                    harfler.append(board_old[y_basl][x_koord])
                    puan=puan+harf_puanlari[board_old[y_basl][x_koord]]
                    #puan.append(tahta_puanlari2[y_basl,x_koord])
                y_basl += 1
        
            #print("puan"+str(puan))
        
            # Join the list to form the final word
            word = "".join(tam_kelime)
            #print("Final word:", word)
            olusan_kelimeler.append(word)
            #print("Puan="+str(puan))
            puanlardizisi.append(puan)
        
            x_koord += 1
            puantoplam=puantoplam+puan
        olusan_kelimeler.append(kelime)
        puantoplam = puantoplam + sum(harf_puanlari[a] for a in kelime)
        #print("tabloya koyduğunuz kelime sonrası oluşan tüm kelimeler şunlardır "+ str(olusan_kelimeler))
        #print("topladığınız toplam puan= "+ str(puantoplam))
        
        
        
    else:
    
        i = 0  # Track position in `kelime`
        olusan_kelimeler=[]
        yler=[]
        xler=[]
        harfler=[]
        puantoplam=0
        puanlardizisi=[]
        for a in kelime:
            x_basl = x_koord
        
            #print("Current letter:", a)
        
            # Look up for the start of the word segment
            x = x_koord - 1
            while x >= 0 and board_old[y_koord][x] != "":
                x -= 1
            x_basl = x + 1
            #print("x_basl=" + str(x_basl))
        
            # Look down for the end of the word segment
            x = x_koord + 1
            while x < len(board_old) and board_old[y_koord][x] != "":
                x += 1
            x_bitis = x - 1
            #print("x_bitiş=" + str(x_bitis))
        
            # Construct the word with letters from `kelime` and existing board letters
            tam_kelime = []
            puan=0

            while x_basl <= x_bitis:
                # Insert from `kelime` if this position corresponds to one of its letters
                if x_basl == x_koord and i < len(kelime):
                    yler.append(y_koord)
                    xler.append(x_basl)
                    harfler.append(kelime[i])
                    tam_kelime.append(kelime[i])
                    puan=puan+harf_puanlari[kelime[i]]
                    #puan.append(tahta_puanlari2[y_koord,x_basl])
                    i += 1  # Move to the next letter in `kelime`
                else:
                    # Otherwise, use the letter from `board_old` if available
                    tam_kelime.append(board_old[y_koord][x_basl])
                    yler.append(y_koord)
                    xler.append(x_basl)
                    harfler.append(board_old[y_koord][x_basl])
                    puan=puan+harf_puanlari[board_old[y_koord][x_basl]]
                x_basl += 1
        
            #print("Constructed letters:", tam_kelime)
            #print("puan"+str(puan))
            # Join the list to form the final word
            word = "".join(tam_kelime)
            #print("Final word:", word)
            olusan_kelimeler.append(word)
            #print("Puan="+str(puan))
            puanlardizisi.append(puan)
        
            y_koord += 1
            puantoplam=puantoplam+puan
        olusan_kelimeler.append(kelime)
        puantoplam = puantoplam + sum(harf_puanlari[a] for a in kelime)

    ### HARF PUANLARININ ÜSTÜNE GELEN INCREMENTAL PUAN HESAPLAMA
    koyulankelimedegeri=sum(harf_puanlari[a] for a in kelime)
    b=0
    ilave=0
    if orientation=="h" or orientation=="H":
        for a in kelime:
            #print(x_koord1)
            #print(y_koord1)
            if tahta_puanlari2[y_koord1,x_koord1]==6:
                ilave=ilave+koyulankelimedegeri*2
                ilave=ilave+puanlardizisi[b]*2
                #tahta_puanlari2[y_koord1,x_koord1]=0                
            elif tahta_puanlari2[y_koord1,x_koord1]==4:
                ilave=ilave+koyulankelimedegeri*1
                ilave=ilave+puanlardizisi[b]*1
                #tahta_puanlari2[y_koord1,x_koord1]=0
            elif tahta_puanlari2[y_koord1,x_koord1]==3:
                ilave=ilave+harf_puanlari[a]*2
                #tahta_puanlari2[y_koord1,x_koord1]=0
            elif tahta_puanlari2[y_koord1,x_koord1]==2:
                ilave=ilave+harf_puanlari[a]
                #tahta_puanlari2[y_koord1,x_koord1]=0
            elif tahta_puanlari2[y_koord1,x_koord1]==25:
                ilave=ilave+25
                
            
            else:
                bos=0
            b=b+1
            x_koord1+=1
    
    else:
        for a in kelime:
            #print(x_koord1)
            #print(y_koord1)
            if tahta_puanlari2[y_koord1,x_koord1]==6:
                ilave=ilave+puanlardizisi[b]*2
                #tahta_puanlari2[y_koord1,x_koord1]=0
            elif tahta_puanlari2[y_koord1,x_koord1]==4:
                ilave=ilave+koyulankelimedegeri*1
                ilave=ilave+puanlardizisi[b]*1
                #tahta_puanlari2[y_koord1,x_koord1]=0
            elif tahta_puanlari2[y_koord1,x_koord1]==3:
                ilave=ilave+harf_puanlari[a]*2
                #tahta_puanlari2[y_koord1,x_koord1]=0
            elif tahta_puanlari2[y_koord1,x_koord1]==2:
                ilave=ilave+harf_puanlari[a]
                #tahta_puanlari2[y_koord1,x_koord1]=0
            elif tahta_puanlari2[y_koord1,x_koord1]==25:
                ilave=ilave+25            
            else:
                bos=0
            y_koord1+=1
                
        
    
    #print(tahta_puanlari2[y_koord1,x_koord1])
        
    #print("ilave puan ="+ str(ilave))
    
    nihai_puan=puantoplam+ilave
    
    
    degerler=list(zip(harfler,(list(zip(xler,yler)))))
    return (olusan_kelimeler , nihai_puan)


#############################################################


def ilk_kelime_yerlestir(kelime,x_koord , y_koord , orientation , board):
    y_koord_perm=y_koord
    x_koord_perm=x_koord
    temp_kelime=[]
    stoktan_dus=[]
    if orientation=="h" or orientation=="H":
        for a in kelime:
            if x_koord<15:
                if board[y_koord][x_koord]=="":
                    temp_kelime.append(a)
                    stoktan_dus.append(a)
                    x_koord+=1
                elif board[y_koord][x_koord]==a:
                    temp_kelime.append(a)
                    x_koord+=1
                else :
                    print("Geçersiz Giriş")
                    break
            else :
                print("sınır dışı")
                temp_kelime=[]
                stoktan_dus=[]
                break
    else:
        for a in kelime:
            if y_koord<15:
                if board[y_koord][x_koord]=="":
                    temp_kelime.append(a)
                    stoktan_dus.append(a)
                    y_koord+=1
                elif board[y_koord][x_koord]==a:
                    temp_kelime.append(a)
                    y_koord+=1
                else :
                    print("Geçersiz Giriş")
                    break
            else:
                print("sınır dışı")
                temp_kelime=[]
                stoktan_dus=[]
                break
    print(temp_kelime)
    print(stoktan_dus)
    
    if orientation=="h" or orientation=="H":
        for a in temp_kelime:
            board[y_koord_perm][x_koord_perm]=a
            x_koord_perm+=1
    else:
        for a in temp_kelime:
            board[y_koord_perm][x_koord_perm]=a
            y_koord_perm+=1
    return(stoktan_dus)
            
            
            
            
#####################################################################


def kelime_yerlestir(kelime,x_koord , y_koord , orientation , board):
    y_koord_perm=y_koord
    x_koord_perm=x_koord
    """
    #BURADA KELİME ETRAFINA SANAL BİR KUTU ÇİZEREK , BU KUTU İÇİNDE ELEMENT OLUP
    #OLMADIĞINA BAKACAĞIZ. EN AZ 1 ELEMEN OLMALI Kİ , KELİMEMİZ BAŞKA BİR
    #KELİMEYE DEĞİYOR OLSUN
    #EĞER GİRİLEN KELİME TABLODAKİ DİĞER KELİMELERE DEĞMİYORSA DEĞİŞKENE 1
    #EĞER GİRİLEN KELİME TABLODAKİ DİĞER KELİMELERLE UYUMLU DEĞİLSE DEĞİŞKENE 2
    #EĞER GİRİLEN KELİME SINIR DIŞINA TAŞIYORSA DEĞİŞKENE 3 DEĞERİ ATANACAK
    #HATA YOKSA , GİRİLEN KELİME İLE HANGİ HARFLERİN KULLANILDIĞI VE STOKTAN DÜŞÜLECEĞİ DÖNÜLECEK
    """
    kontrol_koordinatlar=[]
    if orientation=="h" or orientation=="H":
        for n in range((x_koord) , (x_koord)+len(kelime)):
            kontrol_koordinatlar.append((n, y_koord - 1))
            kontrol_koordinatlar.append((n, y_koord + 1))
        kontrol_koordinatlar.append((x_koord-1,y_koord))
        kontrol_koordinatlar.append((x_koord+len(kelime),y_koord))
    else:
        for n in range((y_koord) , (y_koord)+len(kelime)):
            kontrol_koordinatlar.append((x_koord - 1 , n))
            kontrol_koordinatlar.append((x_koord + 1 , n))
        kontrol_koordinatlar.append((x_koord,y_koord-1))
        kontrol_koordinatlar.append((x_koord,y_koord+len(kelime)))
    filtreli_koordinatlar = [(x, y) for (x, y) in kontrol_koordinatlar \
        if 0 <= x < 15 and 0 <= y < 15]
    #for a in kontrol_koordinatlar:
    #    print(a)
    
    #print(list(kontrol_koordinatlar))
    #print(list(filtreli_koordinatlar))
    checkdigit=0
    for a in filtreli_koordinatlar:
        if board[a[1]][a[0]]=="":
            checkdigit=checkdigit
            #print ("Boş ve "+str(checkdigit))
            #print ("degerA0A1= "+board[a[0]][a[1]])
            #print ("A0= "+str(a[0]))
            #print ("A1= "+str(a[1]))
            #print ("-----------------------------")
        else:
            checkdigit+=1
            #print ("Dolu ve "+str(checkdigit))
    #print("Final Check="+str(checkdigit))
    
    if checkdigit==0:
        #print ("Yanlış konumlama. Kelimenizi tekrar konulayınız")
        return (1)
    else:
        y_koord_perm=y_koord
        x_koord_perm=x_koord
        temp_kelime=[]
        stoktan_dus=[]
        if orientation=="h" or orientation=="H":
            for a in kelime:
                if x_koord<15:
                    if board[y_koord][x_koord]=="":
                        temp_kelime.append(a)
                        stoktan_dus.append(a)
                        x_koord+=1
                    elif board[y_koord][x_koord]==a:
                        temp_kelime.append(a)
                        x_koord+=1
                    else :
                        #print("Geçersiz Giriş")
                        return(2)
                else :
                    #print("sınır dışı")
                    temp_kelime=[]
                    stoktan_dus=[]
                    return (3)
        else:
            for a in kelime:
                if y_koord<15:
                    if board[y_koord][x_koord]=="":
                        temp_kelime.append(a)
                        stoktan_dus.append(a)
                        y_koord+=1
                    elif board[y_koord][x_koord]==a:
                        temp_kelime.append(a)
                        y_koord+=1
                    else :
                        #print("Geçersiz Giriş")
                        return(2)
                else:
                    #print("sınır dışı")
                    temp_kelime=[]
                    stoktan_dus=[]
                    return (3)
        #print(temp_kelime)
        #print(stoktan_dus)
    
    if orientation=="h" or orientation=="H":
        for a in temp_kelime:
            board[y_koord_perm][x_koord_perm]=a
            x_koord_perm+=1
    else:
        for a in temp_kelime:
            board[y_koord_perm][x_koord_perm]=a
            y_koord_perm+=1
    return(stoktan_dus)
 

###############################################################################################

def kelime_yerlestir_new(kelime,x_koord , y_koord , orientation , board):
    y_koord_perm=y_koord
    x_koord_perm=x_koord
    """
    #BURADA KELİME ETRAFINA SANAL BİR KUTU ÇİZEREK , BU KUTU İÇİNDE ELEMENT OLUP
    #OLMADIĞINA BAKACAĞIZ. EN AZ 1 ELEMEN OLMALI Kİ , KELİMEMİZ BAŞKA BİR
    #KELİMEYE DEĞİYOR OLSUN
    #EĞER GİRİLEN KELİME TABLODAKİ DİĞER KELİMELERE DEĞMİYORSA DEĞİŞKENE 1
    #EĞER GİRİLEN KELİME TABLODAKİ DİĞER KELİMELERLE UYUMLU DEĞİLSE DEĞİŞKENE 2
    #EĞER GİRİLEN KELİME SINIR DIŞINA TAŞIYORSA DEĞİŞKENE 3 DEĞERİ ATANACAK
    #HATA YOKSA , GİRİLEN KELİME İLE HANGİ HARFLERİN KULLANILDIĞI VE STOKTAN DÜŞÜLECEĞİ DÖNÜLECEK
    """
    kontrol_koordinatlar=[]
    if orientation=="h" or orientation=="H" :
        for n in range((x_koord) , (x_koord)+len(kelime)):
            kontrol_koordinatlar.append((n, y_koord - 1))
            kontrol_koordinatlar.append((n, y_koord + 1))
        kontrol_koordinatlar.append((x_koord-1,y_koord))
        kontrol_koordinatlar.append((x_koord+len(kelime),y_koord))
    else:
        for n in range((y_koord) , (y_koord)+len(kelime)):
            kontrol_koordinatlar.append((x_koord - 1 , n))
            kontrol_koordinatlar.append((x_koord + 1 , n))
        kontrol_koordinatlar.append((x_koord,y_koord-1))
        kontrol_koordinatlar.append((x_koord,y_koord+len(kelime)))
    filtreli_koordinatlar = [(x, y) for (x, y) in kontrol_koordinatlar \
        if 0 <= x < 15 and 0 <= y < 15]
    #for a in kontrol_koordinatlar:
    #    print(a)
    
    #print(list(kontrol_koordinatlar))
    #print(list(filtreli_koordinatlar))
    checkdigit=0
    for a in filtreli_koordinatlar:
        if board[a[1]][a[0]]=="":
            checkdigit=checkdigit
            #print ("Boş ve "+str(checkdigit))
            #print ("degerA0A1= "+board[a[0]][a[1]])
            #print ("A0= "+str(a[0]))
            #print ("A1= "+str(a[1]))
            #print ("-----------------------------")
        else:
            checkdigit+=1
            #print ("Dolu ve "+str(checkdigit))
    #print("Final Check="+str(checkdigit))
    
    if checkdigit==0:
        #print ("Yanlış konumlama. Kelimenizi tekrar konulayınız")
        return (1)
    else:
        y_koord_perm=y_koord
        x_koord_perm=x_koord
        temp_kelime=[]
        stoktan_dus=[]
        stokmu=[]
        if orientation=="h":
            for a in kelime:
                if x_koord<15:
                    if board[y_koord][x_koord]=="":
                        temp_kelime.append(a)
                        stoktan_dus.append(a)
                        stokmu.append("0")
                        x_koord+=1
                    elif board[y_koord][x_koord]==a:
                        temp_kelime.append(a)
                        stoktan_dus.append(a)
                        stokmu.append("1")
                        x_koord+=1
                    else :
                        #print("Geçersiz Giriş")
                        return(2)
                else :
                    #print("sınır dışı")
                    temp_kelime=[]
                    stoktan_dus=[]
                    return (3)
        else:
            for a in kelime:
                if y_koord<15:
                    if board[y_koord][x_koord]=="":
                        temp_kelime.append(a)
                        stoktan_dus.append(a)
                        stokmu.append("0")
                        y_koord+=1
                    elif board[y_koord][x_koord]==a:
                        temp_kelime.append(a)
                        stoktan_dus.append(a)
                        stokmu.append("1")
                        y_koord+=1
                    else :
                        #print("Geçersiz Giriş")
                        return(2)
                else:
                    #print("sınır dışı")
                    temp_kelime=[]
                    stoktan_dus=[]
                    return (3)
        #print(temp_kelime)
        #print(stoktan_dus)
    
    if orientation=="h" or orientation=="H":
        for a in temp_kelime:
            board[y_koord_perm][x_koord_perm]=a
            x_koord_perm+=1
    else:
        for a in temp_kelime:
            board[y_koord_perm][x_koord_perm]=a
            y_koord_perm+=1
    return(stoktan_dus , stokmu)

    
####################################################################################


###  BU KOD İLE , KOYULAN BİR KELİME SONRASI TABLODA AÇIĞA ÇIKAN YENİ KELİMELER VERİLİYOR. ###
###          AYNI ZAMANDA BU KELİMELERİ OLUŞTURAN HARFLERİN X VE Y KOORDİNATLARI DÖNÜLÜYOR ###
###                     BU KOORDİNATLAR HARF VE KELİME PUANLAMADA KULLANILACAKLAR          ###
###KOG GELİŞTİRİLDİ VE TAHTA KATSAYILARINI DA DİKKATE ALAN ELDE EDİLEN NİHAİ PUAN DÖNÜLÜYOR### 
def kelime_kontrol_new(kelime, x_koord, y_koord, orientation , board_old , harf_puanlari , tahta_puanlari2 , stok , stokindis):
    """    
    ###  BU KOD İLE , KOYULAN BİR KELİME SONRASI TABLODA AÇIĞA ÇIKAN YENİ KELİMELER VERİLİYOR. ###
    ###          AYNI ZAMANDA BU KELİMELERİ OLUŞTURAN HARFLERİN X VE Y KOORDİNATLARI DÖNÜLÜYOR ###
    ###                     BU KOORDİNATLAR HARF VE KELİME PUANLAMADA KULLANILACAKLAR          ###
    ###KOG GELİŞTİRİLDİ VE TAHTA KATSAYILARINI DA DİKKATE ALAN ELDE EDİLEN NİHAİ PUAN DÖNÜLÜYOR### 
    """

    
    x_koord1=x_koord
    y_koord1=y_koord

    if orientation=="h" or orientation=="H" :
    
        i = 0  # Track position in `kelime`
        olusan_kelimeler=[]
        yler=[]
        xler=[]
        harfler=[]
        puantoplam=0
        puanlardizisi=[]
        for q , a in enumerate(kelime):
            if (int(stokindis[q])==0):
                print("Q=" + str(q))
                print("Stokindis=" + str(stokindis[q]))
                print("A=" + str(a))
                #print("----------------------------------------")

                y_basl = y_koord

                #print("Current letter:", a)

                # Look up for the start of the word segment
                y = y_koord - 1
                while y >= 0 and board_old[y][x_koord] != "":
                    y -= 1
                y_basl = y + 1
                #print("y_basl=" + str(y_basl))

                # Look down for the end of the word segment
                y = y_koord + 1
                while y < len(board_old) and board_old[y][x_koord] != "":
                    y += 1
                y_bitis = y - 1
                #print("y_bitiş=" + str(y_bitis))

                # Construct the word with letters from `kelime` and existing board letters
                tam_kelime = []
                puan=0
                #puan=[]
                while y_basl <= y_bitis:
                    # Insert from `kelime` if this position corresponds to one of its letters
                    if y_basl == y_koord and i < len(kelime):
                        yler.append(y_basl)
                        xler.append(x_koord)
                        harfler.append(kelime[i])
                        tam_kelime.append(kelime[i])
                        harf_skor = 0 if i in jokerindis else harf_puanlari[kelime[i]]
                        puan=puan+harf_skor
                        #puan.append(tahta_puanlari2[y_basl,x_koord])
                        i += 1  # Move to the next letter in `kelime`
                    else:
                        # Otherwise, use the letter from `board_old` if available
                        tam_kelime.append(board_old[y_basl][x_koord])
                        yler.append(y_basl)
                        xler.append(x_koord)
                        harfler.append(board_old[y_basl][x_koord])
                        puan=puan+_harf_puani_from_cell(board_old[y_basl][x_koord], harf_puanlari)
                        #puan.append(tahta_puanlari2[y_basl,x_koord])
                    y_basl += 1

                #print("puan"+str(puan))

                # Join the list to form the final word
                word = "".join(tam_kelime)
                #print("Final word:", word)
                olusan_kelimeler.append(word)
                #print("Puan="+str(puan))
                puanlardizisi.append(puan)

            else:
                print("Q=" + str(q))
                print("Stokindis=" + str(stokindis[q]))
                print("A=" + str(a))
                puanlardizisi.append(0)
                olusan_kelimeler.append(a)
                #print("************************************")
                #tam_kelime = []
                #puan=0
                i+=1

            
            print("Oluşan="+ str(olusan_kelimeler))
            print("Puanlar Dizisi="+ str(puanlardizisi))
            print("-----------------------------------")
            x_koord += 1
            puantoplam=puantoplam+puan
        olusan_kelimeler.append(kelime)
        puantoplam = puantoplam + sum(harf_puanlari[a] for a in kelime)
        #print("tabloya koyduğunuz kelime sonrası oluşan tüm kelimeler şunlardır "+ str(olusan_kelimeler))
        #print("topladığınız toplam puan= "+ str(puantoplam))
        
        
        
    else:
    
        i = 0  # Track position in `kelime`
        olusan_kelimeler=[]
        yler=[]
        xler=[]
        harfler=[]
        puantoplam=0
        puanlardizisi=[]
        for q , a in enumerate(kelime):
            if (int(stokindis[q])==0):
                print("Q=" + str(q))
                print("Stokindis=" + str(stokindis[q]))
                print("A=" + str(a))
                #print("----------------------------------------")

                x_basl = x_koord

                #print("Current letter:", a)

                # Look up for the start of the word segment
                x = x_koord - 1
                while x >= 0 and board_old[y_koord][x] != "":
                    x -= 1
                x_basl = x + 1
                #print("x_basl=" + str(x_basl))

                # Look down for the end of the word segment
                x = x_koord + 1
                while x < len(board_old) and board_old[y_koord][x] != "":
                    x += 1
                x_bitis = x - 1
                #print("x_bitiş=" + str(x_bitis))

                # Construct the word with letters from `kelime` and existing board letters
                tam_kelime = []
                puan=0

                while x_basl <= x_bitis:
                    # Insert from `kelime` if this position corresponds to one of its letters
                    if x_basl == x_koord and i < len(kelime):
                        yler.append(y_koord)
                        xler.append(x_basl)
                        harfler.append(kelime[i])
                        tam_kelime.append(kelime[i])
                        harf_skor = 0 if i in jokerindis else harf_puanlari[kelime[i]]
                        puan=puan+harf_skor
                        #puan.append(tahta_puanlari2[y_koord,x_basl])
                        i += 1  # Move to the next letter in `kelime`
                    else:
                        # Otherwise, use the letter from `board_old` if available
                        tam_kelime.append(board_old[y_koord][x_basl])
                        yler.append(y_koord)
                        xler.append(x_basl)
                        harfler.append(board_old[y_koord][x_basl])
                        puan=puan+_harf_puani_from_cell(board_old[y_koord][x_basl], harf_puanlari)
                    x_basl += 1

                #print("Constructed letters:", tam_kelime)
                #print("puan"+str(puan))
                # Join the list to form the final word
                word = "".join(tam_kelime)
                #print("Final word:", word)
                olusan_kelimeler.append(word)
                #print("Puan="+str(puan))
                puanlardizisi.append(puan)
        
            else:
                print("Q=" + str(q))
                print("Stokindis=" + str(stokindis[q]))
                print("A=" + str(a))
                puanlardizisi.append(0)
                #print("****************************")
                #tam_kelime = []
                #puan=0
                olusan_kelimeler.append(a)
                i+=1
            print("oluşan="+ str(olusan_kelimeler))
            print("Puanlar Dizisi="+ str(puanlardizisi))
            print("-----------------------------------")

            y_koord += 1
            puantoplam=puantoplam+puan
        olusan_kelimeler.append(kelime)
        puantoplam = puantoplam + sum(harf_puanlari[a] for a in kelime)

    ### HARF PUANLARININ ÜSTÜNE GELEN INCREMENTAL PUAN HESAPLAMA
    koyulankelimedegeri=sum(harf_puanlari[a] for a in kelime)
    print ("KOYULAN KELİME="+str(koyulankelimedegeri))
    b=0
    ilave=0
    if orientation=="h" or orientation=="H":
        for a in kelime:
            #print(x_koord1)
            #print(y_koord1)
            if tahta_puanlari2[y_koord1,x_koord1]==6:
                ilave=ilave+koyulankelimedegeri*2
                ilave=ilave+puanlardizisi[b]*2
                #tahta_puanlari2[y_koord1,x_koord1]=0
            elif tahta_puanlari2[y_koord1,x_koord1]==4:
                ilave=ilave+koyulankelimedegeri*1
                ilave=ilave+puanlardizisi[b]*1
                #tahta_puanlari2[y_koord1,x_koord1]=0
            elif tahta_puanlari2[y_koord1,x_koord1]==3:
                ilave=ilave+harf_puanlari[a]*2
                #tahta_puanlari2[y_koord1,x_koord1]=0
            elif tahta_puanlari2[y_koord1,x_koord1]==2:
                ilave=ilave+harf_puanlari[a]*2
                #tahta_puanlari2[y_koord1,x_koord1]=0
            elif tahta_puanlari2[y_koord1,x_koord1]==25:
                ilave=ilave+25
            
            else:
                bos=0
            #b=b+1
            #x_koord1+=1
            print("Harf="+str(a))
            print("B="+str(b))
            print("Paunlar dizisi="+str(puanlardizisi[b]))
            print("İlave="+str(ilave))
            print("**************")
            b=b+1
            x_koord1+=1
    
    else:
        for a in kelime:
            #print(x_koord1)
            #print(y_koord1)
            if tahta_puanlari2[y_koord1,x_koord1]==6:
                ilave=ilave+puanlardizisi[b]*2
                #tahta_puanlari2[y_koord1,x_koord1]=0
            elif tahta_puanlari2[y_koord1,x_koord1]==4:
                ilave=ilave+koyulankelimedegeri*1
                ilave=ilave+puanlardizisi[b]*1
                #tahta_puanlari2[y_koord1,x_koord1]=0
            elif tahta_puanlari2[y_koord1,x_koord1]==3:
                ilave=ilave+harf_puanlari[a]*2
                #tahta_puanlari2[y_koord1,x_koord1]=0
            elif tahta_puanlari2[y_koord1,x_koord1]==2:
                ilave=ilave+harf_puanlari[a]*2
                #tahta_puanlari2[y_koord1,x_koord1]=0
            elif tahta_puanlari2[y_koord1,x_koord1]==25:
                ilave=ilave+25


            
            else:
                bos=0
            print("Harf="+str(a))
            print("B="+str(b))
            print("Paunlar dizisi="+str(puanlardizisi[b]))
            print("İlave="+str(ilave))
            print("***********************")
            b=b+1
            y_koord1+=1
                
        
    
    #print(tahta_puanlari2[y_koord1,x_koord1])
        
    #print("ilave puan ="+ str(ilave))
    
    nihai_puan=puantoplam+ilave
    
  
    degerler=list(zip(harfler,(list(zip(xler,yler)))))
    return (olusan_kelimeler , nihai_puan)


####################################################################################

def kelime_kontrol_final(kelime, x_koord, y_koord, orientation , board_old , harf_puanlari , tahta_puanlari2 , stok , stokindis, jokerindis=None):
    """    
    ###  BU KOD İLE , KOYULAN BİR KELİME SONRASI TABLODA AÇIĞA ÇIKAN YENİ KELİMELER VERİLİYOR. ###
    ###          AYNI ZAMANDA BU KELİMELERİ OLUŞTURAN HARFLERİN X VE Y KOORDİNATLARI DÖNÜLÜYOR ###
    ###                     BU KOORDİNATLAR HARF VE KELİME PUANLAMADA KULLANILACAKLAR          ###
    ###KOG GELİŞTİRİLDİ VE TAHTA KATSAYILARINI DA DİKKATE ALAN ELDE EDİLEN NİHAİ PUAN DÖNÜLÜYOR### 
    """

    
    kelime = _normalize_input_word(kelime)

    x_koord1=x_koord
    y_koord1=y_koord
    jokerindis = set(jokerindis or [])
    if orientation=="h" or orientation=="H":
    
        i = 0  # Track position in `kelime`
        olusan_kelimeler=[]
        yler=[]
        xler=[]
        harfler=[]        
        puanlardizisi=[]
        for q , a in enumerate(kelime):
            if (int(stokindis[q])==0):
                y_basl = y_koord

                # Look up for the start of the word segment
                y = y_koord - 1
                while y >= 0 and board_old[y][x_koord] != "":
                    y -= 1
                y_basl = y + 1
                
                # Look down for the end of the word segment
                y = y_koord + 1
                while y < len(board_old) and board_old[y][x_koord] != "":
                    y += 1
                y_bitis = y - 1
                
                # Construct the word with letters from `kelime` and existing board letters
                tam_kelime = []
                puan=0               
                while y_basl <= y_bitis:
                    # Insert from `kelime` if this position corresponds to one of its letters
                    if y_basl == y_koord and i < len(kelime):
                        yler.append(y_basl)
                        xler.append(x_koord)
                        harfler.append(kelime[i])
                        tam_kelime.append(kelime[i])
                        harf_skor = 0 if i in jokerindis else harf_puanlari.get(kelime[i], 0)
                        puan=puan+harf_skor                       
                        i += 1  # Move to the next letter in `kelime`
                    else:
                        # Otherwise, use the letter from `board_old` if available
                        tam_kelime.append(board_old[y_basl][x_koord])
                        yler.append(y_basl)
                        xler.append(x_koord)
                        harfler.append(board_old[y_basl][x_koord])
                        puan=puan+_harf_puani_from_cell(board_old[y_basl][x_koord], harf_puanlari)                        
                    y_basl += 1

                # Join the list to form the final word
                word = "".join(tam_kelime)
                olusan_kelimeler.append(word)                
                puanlardizisi.append(puan)

            else:
                ##print("Q=" + str(q))
                ##print("Stokindis=" + str(stokindis[q]))
                ##print("A=" + str(a))
                puanlardizisi.append(0)
                olusan_kelimeler.append(a)                
                i+=1

            
            ##print("Oluşan="+ str(olusan_kelimeler))
            ##print("Puanlar Dizisi="+ str(puanlardizisi))
            ##print("-----------------------------------")
            x_koord += 1
        olusan_kelimeler.append(kelime)
        
        
        
    else:
    
        i = 0  # Track position in `kelime`
        olusan_kelimeler=[]
        yler=[]
        xler=[]
        harfler=[]        
        puanlardizisi=[]
        for q , a in enumerate(kelime):
            if (int(stokindis[q])==0):
              

                x_basl = x_koord

                # Look up for the start of the word segment
                x = x_koord - 1
                while x >= 0 and board_old[y_koord][x] != "":
                    x -= 1
                x_basl = x + 1
                
                # Look down for the end of the word segment
                x = x_koord + 1
                while x < len(board_old) and board_old[y_koord][x] != "":
                    x += 1
                x_bitis = x - 1
                
                # Construct the word with letters from `kelime` and existing board letters
                tam_kelime = []
                puan=0
                while x_basl <= x_bitis:
                    # Insert from `kelime` if this position corresponds to one of its letters
                    if x_basl == x_koord and i < len(kelime):
                        yler.append(y_koord)
                        xler.append(x_basl)
                        harfler.append(kelime[i])
                        tam_kelime.append(kelime[i])
                        harf_skor = 0 if i in jokerindis else harf_puanlari.get(kelime[i], 0)
                        puan=puan+harf_skor
                        i += 1  # Move to the next letter in `kelime`
                    else:
                        # Otherwise, use the letter from `board_old` if available
                        tam_kelime.append(board_old[y_koord][x_basl])
                        yler.append(y_koord)
                        xler.append(x_basl)
                        harfler.append(board_old[y_koord][x_basl])
                        puan=puan+_harf_puani_from_cell(board_old[y_koord][x_basl], harf_puanlari)
                    x_basl += 1
                # Join the list to form the final word
                word = "".join(tam_kelime)
                olusan_kelimeler.append(word)
                puanlardizisi.append(puan)
        
            else:
                puanlardizisi.append(0)
                olusan_kelimeler.append(a)
                i+=1
            
            y_koord += 1            
        olusan_kelimeler.append(kelime)
        
    puanlardizisi.append(sum((0 if i in jokerindis else harf_puanlari.get(a, 0)) for i, a in enumerate(kelime)))
    
    
    carpan=1
    kelime_deger=sum((0 if i in jokerindis else harf_puanlari.get(a, 0)) for i, a in enumerate(kelime))
    alternatif_kelimeler_puan=0
    if orientation=="h" or orientation=="H":
        for q,a in enumerate(kelime):
            if tahta_puanlari2[y_koord1,x_koord1]==6:
                carpan=carpan*3
                if len(str(olusan_kelimeler[q]))!=1:     #EĞER KOYULAN HARF , TEK HARFLİ ALTERNATİF KELİME OLARAK YER ALMIYORSA
                    alternatif_kelimeler_puan=alternatif_kelimeler_puan+puanlardizisi[q]*3
                else:    #EĞER KOYULAN HARF , TEK HARFLİ ALTERNATİF KELİME OLARAK YER ALIYORSA
                    alternatif_kelimeler_puan=alternatif_kelimeler_puan
                #tahta_puanlari2[y_koord1,x_koord1]=0
            elif tahta_puanlari2[y_koord1,x_koord1]==4:
                carpan=carpan*2
                if len(str(olusan_kelimeler[q]))!=1:     #EĞER KOYULAN HARF , TEK HARFLİ ALTERNATİF KELİME OLARAK YER ALMIYORSA
                    alternatif_kelimeler_puan=alternatif_kelimeler_puan+puanlardizisi[q]*2
                else:    #EĞER KOYULAN HARF , TEK HARFLİ ALTERNATİF KELİME OLARAK YER ALIYORSA
                    alternatif_kelimeler_puan=alternatif_kelimeler_puan
                #tahta_puanlari2[y_koord1,x_koord1]=0
            elif tahta_puanlari2[y_koord1,x_koord1]==3:
                harf_baz = 0 if q in jokerindis else harf_puanlari.get(a, 0)
                kelime_deger=kelime_deger+harf_baz*2
                if len(str(olusan_kelimeler[q]))!=1:     #EĞER KOYULAN HARF , TEK HARFLİ ALTERNATİF KELİME OLARAK YER ALMIYORSA
                    alternatif_kelimeler_puan=alternatif_kelimeler_puan+puanlardizisi[q]+harf_baz*2
                else:    #EĞER KOYULAN HARF , TEK HARFLİ ALTERNATİF KELİME OLARAK YER ALIYORSA
                    alternatif_kelimeler_puan=alternatif_kelimeler_puan
                #tahta_puanlari2[y_koord1,x_koord1]=0
            elif tahta_puanlari2[y_koord1,x_koord1]==2:
                harf_baz = 0 if q in jokerindis else harf_puanlari.get(a, 0)
                kelime_deger=kelime_deger+harf_baz*1
                if len(str(olusan_kelimeler[q]))!=1:     #EĞER KOYULAN HARF , TEK HARFLİ ALTERNATİF KELİME OLARAK YER ALMIYORSA
                    alternatif_kelimeler_puan=alternatif_kelimeler_puan+puanlardizisi[q]+harf_baz*1
                else:    #EĞER KOYULAN HARF , TEK HARFLİ ALTERNATİF KELİME OLARAK YER ALIYORSA
                    alternatif_kelimeler_puan=alternatif_kelimeler_puan
                #tahta_puanlari2[y_koord1,x_koord1]=0
            elif tahta_puanlari2[y_koord1,x_koord1]==25:
                kelime_deger=kelime_deger+25


            
            else:
                kelime_deger=kelime_deger
                if len(str(olusan_kelimeler[q]))!=1:     #EĞER KOYULAN HARF , TEK HARFLİ ALTERNATİF KELİME OLARAK YER ALMIYORSA
                    alternatif_kelimeler_puan=alternatif_kelimeler_puan+puanlardizisi[q]
                else:    #EĞER KOYULAN HARF , TEK HARFLİ ALTERNATİF KELİME OLARAK YER ALIYORSA
                    alternatif_kelimeler_puan=alternatif_kelimeler_puan
            '''
            print("kelime_deger="+str(kelime_deger))
            print("Tahta işareti="+str(tahta_puanlari2[y_koord1,x_koord1]))
            print("Y="+str(y_koord1))
            print("X="+str(x_koord1))
            print("---------------------------------")
            '''
        
            x_koord1+=1
            
            
            
    else:
        for q,a in enumerate(kelime):
            if tahta_puanlari2[y_koord1,x_koord1]==6:
                carpan=carpan*3
                if len(str(olusan_kelimeler[q]))!=1:     #EĞER KOYULAN HARF , TEK HARFLİ ALTERNATİF KELİME OLARAK YER ALMIYORSA
                    alternatif_kelimeler_puan=alternatif_kelimeler_puan+puanlardizisi[q]*3
                else:    #EĞER KOYULAN HARF , TEK HARFLİ ALTERNATİF KELİME OLARAK YER ALIYORSA
                    alternatif_kelimeler_puan=alternatif_kelimeler_puan
                #tahta_puanlari2[y_koord1,x_koord1]=0
            elif tahta_puanlari2[y_koord1,x_koord1]==4:
                carpan=carpan*2
                if len(str(olusan_kelimeler[q]))!=1:     #EĞER KOYULAN HARF , TEK HARFLİ ALTERNATİF KELİME OLARAK YER ALMIYORSA
                    alternatif_kelimeler_puan=alternatif_kelimeler_puan+puanlardizisi[q]*2
                else:    #EĞER KOYULAN HARF , TEK HARFLİ ALTERNATİF KELİME OLARAK YER ALIYORSA
                    alternatif_kelimeler_puan=alternatif_kelimeler_puan
                #tahta_puanlari2[y_koord1,x_koord1]=0
            elif tahta_puanlari2[y_koord1,x_koord1]==3:
                harf_baz = 0 if q in jokerindis else harf_puanlari.get(a, 0)
                kelime_deger=kelime_deger+harf_baz*2
                if len(str(olusan_kelimeler[q]))!=1:     #EĞER KOYULAN HARF , TEK HARFLİ ALTERNATİF KELİME OLARAK YER ALMIYORSA
                    alternatif_kelimeler_puan=alternatif_kelimeler_puan+puanlardizisi[q]+harf_baz*2
                else:    #EĞER KOYULAN HARF , TEK HARFLİ ALTERNATİF KELİME OLARAK YER ALIYORSA
                    alternatif_kelimeler_puan=alternatif_kelimeler_puan
                #tahta_puanlari2[y_koord1,x_koord1]=0
            elif tahta_puanlari2[y_koord1,x_koord1]==2:
                harf_baz = 0 if q in jokerindis else harf_puanlari.get(a, 0)
                kelime_deger=kelime_deger+harf_baz*1
                if len(str(olusan_kelimeler[q]))!=1:     #EĞER KOYULAN HARF , TEK HARFLİ ALTERNATİF KELİME OLARAK YER ALMIYORSA
                    alternatif_kelimeler_puan=alternatif_kelimeler_puan+puanlardizisi[q]+harf_baz*1
                else:    #EĞER KOYULAN HARF , TEK HARFLİ ALTERNATİF KELİME OLARAK YER ALIYORSA
                    alternatif_kelimeler_puan=alternatif_kelimeler_puan
                #tahta_puanlari2[y_koord1,x_koord1]=0
            elif tahta_puanlari2[y_koord1,x_koord1]==25:
                kelime_deger=kelime_deger+25            
            else:
                kelime_deger=kelime_deger
                if len(str(olusan_kelimeler[q]))!=1:     #EĞER KOYULAN HARF , TEK HARFLİ ALTERNATİF KELİME OLARAK YER ALMIYORSA
                    alternatif_kelimeler_puan=alternatif_kelimeler_puan+puanlardizisi[q]
                else:    #EĞER KOYULAN HARF , TEK HARFLİ ALTERNATİF KELİME OLARAK YER ALIYORSA
                    alternatif_kelimeler_puan=alternatif_kelimeler_puan
            '''
            print("kelime_deger="+str(kelime_deger))
            print("Tahta işareti="+str(tahta_puanlari2[y_koord1,x_koord1]))
            print("Y="+str(y_koord1))
            print("X="+str(x_koord1))
            print("**********************************")
            '''
            y_koord1+=1

    kelime_deger=kelime_deger*carpan
        
        
        
    
    
    
    
    
    
    
    final_puan=kelime_deger+alternatif_kelimeler_puan
    
    
    
    return (olusan_kelimeler , puanlardizisi , final_puan , tahta_puanlari2)


#######################################################################
def kelime_yerlestir_ve_puanla(kelime, x_koord, y_koord, orientation, board,tahta_puanlari2,sozluk):
    import copy
    import numpy as np
    board_old = copy.deepcopy(board)
    harf_puanlari = {
    'A': 1, 'B': 3, 'C': 4, 'Ç': 4, 'D': 3, 'E': 1, 'F': 7, 'G': 5, 'Ğ': 8,
    'H': 5, 'I': 2, 'İ': 1, 'J': 10, 'K': 1, 'L': 1, 'M': 2, 'N': 1, 'O': 2,
    'Ö': 7, 'P': 5, 'R': 1, 'S': 2, 'Ş': 4, 'T': 1, 'U': 2, 'Ü': 3, 'V': 7,
    'Y': 3, 'Z': 4
    }
    
       
    # --- ORIENTATION NORMALİZASYONU ---
    ori = str(orientation).strip().lower()
    yatay_mi = ori.startswith("h")

    # Harf puanları ve tahta çarpanları sabit olarak kullanılacak
    #from kelimelik_engine1 import harf_puanlari, tahta_puanlari2

    # --- Yerleşim Kontrolü ---
    kontrol_koordinatlar = []
    if yatay_mi:
        if x_koord + len(kelime) > 15:
            return 3  # sınır dışı
        for n in range(x_koord, x_koord + len(kelime)):
            kontrol_koordinatlar.append((n, y_koord - 1))
            kontrol_koordinatlar.append((n, y_koord + 1))
        kontrol_koordinatlar += [(x_koord - 1, y_koord), (x_koord + len(kelime), y_koord)]
    else:
        if y_koord + len(kelime) > 15:
            return 3  # sınır dışı
        for n in range(y_koord, y_koord + len(kelime)):
            kontrol_koordinatlar.append((x_koord - 1, n))
            kontrol_koordinatlar.append((x_koord + 1, n))
        kontrol_koordinatlar += [(x_koord, y_koord - 1), (x_koord, y_koord + len(kelime))]

    kontrol_koordinatlar = [(x, y) for x, y in kontrol_koordinatlar if 0 <= x < 15 and 0 <= y < 15]
    checkdigit = any(board[y][x] != "" for x, y in kontrol_koordinatlar)
    if not checkdigit and all(board[y_koord][x_koord + i] == "" if yatay_mi else board[y_koord + i][x_koord] == "" for i in range(len(kelime))):
        return 1  # başka kelimeye değmiyor

    # --- Yerleştirme ve stok belirleme ---
    stoktan_dus = []
    stokindis = []
    x, y = x_koord, y_koord
    for harf in kelime:
        if board[y][x] == "":
            stoktan_dus.append(harf)
            stokindis.append("0")
        elif _hucre_harf_eslesiyor_mu(board[y][x], harf):
            stokindis.append("1")
        else:
            return 2  # çakışma var
        if yatay_mi:
            x += 1
        else:
            y += 1

    # --- Harfleri geçici olarak board'a yerleştir ---
    x, y = x_koord, y_koord
    for i, harf in enumerate(kelime):
        if stokindis[i] == "0":
            board[y][x] = harf
        if yatay_mi:
            x += 1
        else:
            y += 1

    # --- kelime_kontrol_final işlevini çağır ---
    from kelimelik_engine1 import kelime_kontrol_final
    olusan_kelimeler, puanlardizisi, toplam_puan , tahta_puanlari2 = kelime_kontrol_final(
        kelime, x_koord, y_koord, orientation, board_old, harf_puanlari, tahta_puanlari2, stoktan_dus, stokindis
    )

    # Her kelime için sözlükte olup olmadığını kontrol et
    kelime_durumlari = [(k, k in sozluk) for k in olusan_kelimeler]

    # Tüm kelimeler sözlükte mi?
    olusan_kelimeler_sozlukte_var_mi = all(durum for (_, durum) in kelime_durumlari)


    return {
        "kelimeler": olusan_kelimeler,
        "kelime_durumlari": kelime_durumlari,
        "puan": toplam_puan,
        "oynanan_kelime_gosterim": _normalize_input_word(kelime),
        "board": board,
        "stoktan_dus": stoktan_dus,
        "yeni_tahta_puanlari": tahta_puanlari2,
        "gecerli": olusan_kelimeler_sozlukte_var_mi
    }

##################################################

def find_scrabble_words3(board):
    '''''
    Bu fonksiyon ile , verilen bir tahtadaki soldan sağa ve yukarıdan aşağı olan tüm kelimecikler dökülmekte

    '''''
    import numpy as np
    words = []
    
    # Dikey tarama (yukarıdan aşağı)
    for col in range(15):
        column = board[:, col]
        i = 0
        while i < 15:
            if column[i] != ' ':
                if i == 0 or column[i-1] == ' ':
                    j = i
                    while j < 15 and column[j] != ' ':
                        j += 1
                    word = ''.join(column[i:j])
                    if len(word) > 0:  # Boş string kontrolü
                        words.append(word)
                    i = j
                else:
                    i += 1
            else:
                i += 1
                
    # Yatay tarama (soldan sağa)
    for row in range(15):
        row_data = board[row, :]
        i = 0
        while i < 15:
            if row_data[i] != ' ':
                if i == 0 or row_data[i-1] == ' ':
                    j = i
                    while j < 15 and row_data[j] != ' ':
                        j += 1
                    word = ''.join(row_data[i:j])
                    if len(word) > 0:  # Boş string kontrolü
                        words.append(word)
                    i = j
                else:
                    i += 1
            else:
                i += 1
                
    return np.array(words)

###########################################
'''
def filter_scrabble_dictionary(sozluk, words):
    
    """
    Bu fonksiyonla verilen 2 harf ve daha büyük kelime gruplarını içeren , sözlükteki tüm kelimeler dökülür.
    Güncellenmiş Filtreleme Fonksiyonu
    
    1. Tüm kelimeleri büyük harfe çevirir (case-insensitive)
    2. Sadece 2+ harfli kelimeleri dikkate alır
    3. Kelimeciklerin sözlükte olma şartını kaldırır
    4. İçinde herhangi bir kelimecik geçen tüm kelimeleri tutar
    
    Parametreler:
    sozluk (numpy array/list): Orijinal sözlük
    words (numpy array): Tahtadaki kelimecikler
    
    Dönüş Değeri:
    numpy array: Filtrelenmiş sözlük (büyük harf, unique, sorted)
    """
    import numpy as np
    # 1. Büyük harfe çevirme ve temizleme işlemleri
    words_upper = np.char.upper(words).astype(str)
    sozluk_upper = np.char.upper(sozluk).astype(str)
    
    # 2. Geçerli kelimecikleri hazırla (en az 2 harfli)
    valid_parts = [part for part in np.unique(words_upper) if len(part) >= 1]
    
    # 3. Sözlüğü temizle (en az 2 harfli)
    clean_sozluk = [kelime for kelime in np.unique(sozluk_upper) if len(kelime) >= 1]
    
    # 4. Filtreleme işlemi
    filtered = []
    for kelime in clean_sozluk:
        # Kelimeciklerden herhangi birini içeriyor mu?
        if any(part in kelime for part in valid_parts):
            filtered.append(kelime)
    
    # 5. Sonuçları sırala ve unique yap
    return np.sort(np.unique(filtered))
'''
#####################################################

def filter_scrabble_dictionary_uzun1(sozluk, words):
    
    """
    Bu fonksiyonla verilen 1 harfli kelime gruplarını içeren , sözlükteki tüm kelimeler dökülür.
    Güncellenmiş Filtreleme Fonksiyonu
    
    1. Tüm kelimeleri büyük harfe çevirir (case-insensitive)
    2. Sadece 2+ harfli kelimeleri dikkate alır
    3. Kelimeciklerin sözlükte olma şartını kaldırır
    4. İçinde herhangi bir kelimecik geçen tüm kelimeleri tutar
    
    Parametreler:
    sozluk (numpy array/list): Orijinal sözlük
    words (numpy array): Tahtadaki kelimecikler
    
    Dönüş Değeri:
    numpy array: Filtrelenmiş sözlük (büyük harf, unique, sorted)
    """
    import numpy as np
    # 1. Büyük harfe çevirme ve temizleme işlemleri
    words_upper = np.char.upper(words).astype(str)
    sozluk_upper = np.char.upper(sozluk).astype(str)
    
    # 2. Geçerli kelimecikleri hazırla (en az 2 harfli)
    valid_parts = [part for part in np.unique(words_upper) if len(part) == 1]
    
    # 3. Sözlüğü temizle (en az 2 harfli)
    clean_sozluk = [kelime for kelime in np.unique(sozluk_upper) if len(kelime) == 1]
    
    # 4. Filtreleme işlemi
    filtered = []
    for kelime in clean_sozluk:
        # Kelimeciklerden herhangi birini içeriyor mu?
        if any(part in kelime for part in valid_parts):
            filtered.append(kelime)
    
    # 5. Sonuçları sırala ve unique yap
    return np.sort(np.unique(filtered))

#############################################################################
'''
def aday_kelime_yerlestir(tahta, aday_kelime):
    """
    Koda tahta ve aday kelimeyi verdiğinizde , aday kelimenin tahtta yerleşebileceği tüm pozisyonları size veriyor
    Ancak herhangi bir puanlama veya sözlük kontrolü yapmıyor. Bu kod kelim ekontrol final'in inputu olacaktır
    """
    boyut = len(tahta)
    kelime_uzunluk = len(aday_kelime)
    gecerli_pozisyonlar = []
    
    for yon in ['horizontal', 'vertical']:
        for y in range(boyut):
            for x in range(boyut):
                # Kelimenin sığma kontrolü (tahta sınırları dahilinde mi?)
                if (yon == 'horizontal' and x + kelime_uzunluk > boyut) or \
                   (yon == 'vertical' and y + kelime_uzunluk > boyut):
                    continue
                
                cakisma_var = False
                cakisamayan_harfler = 0
                komsuluk_var = False
                
                for i in range(kelime_uzunluk):
                    tx = x + (i if yon == 'horizontal' else 0)
                    ty = y + (i if yon == 'vertical' else 0)
                    
                    # Çakışma kontrolü
                    if tahta[ty][tx] != '':
                        if tahta[ty][tx] != aday_kelime[i]:
                            cakisma_var = True
                            break
                        cakisamayan_harfler += 1
                    else:
                        # Komşuluk kontrolü (tüm harfler için dik yönde)
                        if yon == 'horizontal':
                            # Yatay kelime: üst/alt veya aynı satırda yan yana harf
                            if (ty > 0 and tahta[ty-1][tx] != '') or (ty < boyut-1 and tahta[ty+1][tx] != ''):
                                komsuluk_var = True
                            # Ayrıca, yatay kelimenin sol/sağ uçlarını kontrol et
                            if i == 0 and tx > 0 and tahta[ty][tx-1] != '':
                                komsuluk_var = True
                            if i == kelime_uzunluk-1 and tx < boyut-1 and tahta[ty][tx+1] != '':
                                komsuluk_var = True
                        else:
                            # Dikey kelime: sol/sağ veya aynı sütunda yukarı/aşağı
                            if (tx > 0 and tahta[ty][tx-1] != '') or (tx < boyut-1 and tahta[ty][tx+1] != ''):
                                komsuluk_var = True
                            # Ayrıca, dikey kelimenin üst/alt uçlarını kontrol et
                            if i == 0 and ty > 0 and tahta[ty-1][tx] != '':
                                komsuluk_var = True
                            if i == kelime_uzunluk-1 and ty < boyut-1 and tahta[ty+1][tx] != '':
                                komsuluk_var = True
                
                if cakisma_var:
                    continue
                
                # En az 1 çakışma VEYA komşuluk
                if cakisamayan_harfler == 0 and not komsuluk_var:
                    continue
                
                gecerli_pozisyonlar.append((x, y, yon))
    
    return gecerli_pozisyonlar
'''
###########################################################################



def _format_played_word_with_jokers(orijinal_kelime, stokindis, joker_stok_indisleri):
    """Oynanan kelimede joker ile konan harfleri *H şeklinde işaretler."""
    out = []
    yeni_idx = 0
    joker_set = set(joker_stok_indisleri or [])
    for i, ch in enumerate(orijinal_kelime):
        up = str(ch).upper()
        if i < len(stokindis) and stokindis[i] == "0":
            if yeni_idx in joker_set:
                out.append(f"*{up}")
            else:
                out.append(up)
            yeni_idx += 1
        else:
            out.append(up)
    return "".join(out)

def _format_board_cell(cell, use_color=True):
    """Joker hücrelerini (*A) şeklinde ayırt ederek gösterir."""
    if cell == "" or cell is None:
        return " "

    if isinstance(cell, str) and cell.isalpha():
        if _is_joker_cell(cell):
            txt = f"*{cell.upper()}"
            return f"\033[95m{txt}\033[0m" if use_color else txt

        txt = cell.upper()
        return f"\033[93m{txt}\033[0m" if use_color else txt

    txt = str(cell)
    return f"\033[97m{txt}\033[0m" if use_color else txt


def print_board2(board):
    print("     " + "  ".join([f"{i:2}" for i in range(15)]))  # Sütun numaraları
    print("   +" + "---+" * 15)
    for y, row in enumerate(board):
        row_str = " | ".join([_format_board_cell(cell, use_color=False) for cell in row])
        print(f"{y:2} | {row_str} |")
        print("   +" + "---+" * 15)
####################################################################

def print_board(board):
    print("     " + "  ".join([f"{i:2}" for i in range(15)]))
    print("   +" + "---+" * 15)
    for y, row in enumerate(board):
        row_str = [_format_board_cell(cell, use_color=True) for cell in row]
        print(f"{y:2} | " + " | ".join(row_str) + " |")
        print("   +" + "---+" * 15)


##################################################

def extract_words(board):
    """
    Bu kod, tahtaya soldan sağa ve yukarıdan aşağıya bakar,
    karşısına gelen tüm harf veya harf gruplarını, başlangıç koordinatları ve yön bilgisiyle birlikte döner.
    """
    #import numpy as np
    #board=np.array(board)
    size = board.shape[0]
    words = []

    # Soldan sağa tarama (satırlar)
    for y in range(size):
        word = ''
        start_x = None
        for x in range(size):
            if board[y, x] != '':
                if word == '':
                    start_x = x
                word += _normalize_cell_letter(board[y, x])
            else:
                if len(word) >= 1:
                    words.append((word, (start_x, y), 'horizontal'))
                word = ''
                start_x = None
        if len(word) >= 1:
            words.append((word, (start_x, y), 'horizontal'))

    # Yukarıdan aşağıya tarama (sütunlar)
    for x in range(size):
        word = ''
        start_y = None
        for y in range(size):
            if board[y, x] != '':
                if word == '':
                    start_y = y
                word += _normalize_cell_letter(board[y, x])
            else:
                if len(word) >= 1:
                    words.append((word, (x, start_y), 'vertical'))
                word = ''
                start_y = None
        if len(word) >= 1:
            words.append((word, (x, start_y), 'vertical'))

    return words

############################################################
'''
from itertools import permutations
from collections import Counter

def find_possible_words(available_letters, word_parts, dictionary):
    possible_words = set()
    available_lower = available_letters.lower()
    available_counts = Counter(available_lower)
    dictionary = {word.lower() for word in dictionary}

    for part in word_parts:
        part_lower = str(part).lower()
        max_extra_letters = len(available_lower)

        for pre_len in range(max_extra_letters + 1):
            for pre in permutations(available_lower, pre_len):
                pre_counter = Counter(pre)
                if not pre_counter <= available_counts:
                    continue

                remaining = list((available_counts - pre_counter).elements())
                for post_len in range(max_extra_letters - pre_len + 1):
                    for post in permutations(remaining, post_len):
                        post_counter = Counter(post)
                        total_counter = pre_counter + post_counter
                        if not total_counter <= available_counts:
                            continue

                        candidate = ''.join(pre) + part_lower + ''.join(post)
                        if candidate in dictionary:
                            possible_words.add(candidate)


        if part_lower in dictionary:
            possible_words.add(part_lower)

    return sorted(possible_words)
'''
##############################################3
'''
from itertools import permutations
from collections import Counter

def find_possible_words_and_orientations(available_letters, word_parts, dictionary):
    possible_words = set()
    available_lower = available_letters.lower()
    available_counts = Counter(available_lower)
    dictionary = {word.lower() for word in dictionary}

    for part, orientation in word_parts:
        part_lower = str(part).lower()
        max_extra_letters = len(available_lower)

        for pre_len in range(max_extra_letters + 1):
            for pre in permutations(available_lower, pre_len):
                pre_counter = Counter(pre)
                if not pre_counter <= available_counts:
                    continue

                remaining = list((available_counts - pre_counter).elements())
                for post_len in range(max_extra_letters - pre_len + 1):
                    for post in permutations(remaining, post_len):
                        post_counter = Counter(post)
                        total_counter = pre_counter + post_counter
                        if not total_counter <= available_counts:
                            continue

                        candidate = ''.join(pre) + part_lower + ''.join(post)
                        if candidate in dictionary:
                            possible_words.add((candidate, orientation))

        if part_lower in dictionary:
            possible_words.add((part_lower, orientation))

    return sorted(possible_words)
'''
########################################################
from itertools import permutations
from collections import Counter

def find_possible_words_and_orientations2(available_letters, word_parts, dictionary):
    """
    Girdilerin ve çıktının tamamen büyük harfli Türkçe karakterlerle olduğu versiyon.
    """
    possible_words = set()
    available_counts = Counter(available_letters)  # Büyük harflerle sayım yapılır
    dictionary = {word for word in dictionary}     # Sözlük zaten büyük harfli olarak varsayılır

    for part, orientation in word_parts:
        max_extra_letters = len(available_letters)

        for pre_len in range(max_extra_letters + 1):
            for pre in permutations(available_letters, pre_len):
                pre_counter = Counter(pre)
                if not pre_counter <= available_counts:
                    continue

                remaining = list((available_counts - pre_counter).elements())
                for post_len in range(max_extra_letters - pre_len + 1):
                    for post in permutations(remaining, post_len):
                        post_counter = Counter(post)
                        total_counter = pre_counter + post_counter
                        if not total_counter <= available_counts:
                            continue

                        candidate = ''.join(pre) + part + ''.join(post)
                        if candidate in dictionary:
                            possible_words.add((candidate, orientation))

        # Parça doğrudan sözlükte varsa
        if part in dictionary:
            possible_words.add((part, orientation))

    return sorted(possible_words)

####################################################

from itertools import permutations

def find_valid_words_from_available_with_orientations(available_letters, dictionary):
    from collections import Counter
    valid_words = set()

    rack_counter, joker_count = _parse_rack_and_jokers(available_letters)
    max_len = sum(rack_counter.values()) + joker_count

    for word in dictionary:
        w = str(word).upper()
        if len(w) < 2 or len(w) > max_len:
            continue
        if _kelime_can_be_formed_with_jokers(w, rack_counter, joker_count):
            valid_words.add((w, "horizontal"))
            valid_words.add((w, "vertical"))

    return sorted(valid_words)

###########################################################

def aday_kelime_yerlestir(tahta, aday_kelimeler):
    """
    Girdi: aday_kelimeler -> [('KELIME', 'horizontal'), ('KELIME2', 'vertical'), ...]
    Çıktı: [(kelime, x, y, orientation)] - Belirtilen yöndeki geçerli yerleşim pozisyonları
    """
    boyut = len(tahta)
    gecerli_pozisyonlar = []

    for kelime, yon in aday_kelimeler:
        kelime_uzunluk = len(kelime)

        for y in range(boyut):
            for x in range(boyut):
                if (yon == 'horizontal' and x + kelime_uzunluk > boyut) or \
                   (yon == 'vertical' and y + kelime_uzunluk > boyut):
                    continue

                cakisma_var = False
                cakisamayan_harfler = 0
                komsuluk_var = False

                for i in range(kelime_uzunluk):
                    tx = x + (i if yon == 'horizontal' else 0)
                    ty = y + (i if yon == 'vertical' else 0)

                    if tahta[ty][tx] != '':
                        if not _hucre_harf_eslesiyor_mu(tahta[ty][tx], kelime[i]):
                            cakisma_var = True
                            break
                        cakisamayan_harfler += 1
                    else:
                        if yon == 'horizontal':
                            if (ty > 0 and tahta[ty-1][tx] != '') or (ty < boyut-1 and tahta[ty+1][tx] != ''):
                                komsuluk_var = True
                            if i == 0 and tx > 0 and tahta[ty][tx-1] != '':
                                komsuluk_var = True
                            if i == kelime_uzunluk-1 and tx < boyut-1 and tahta[ty][tx+1] != '':
                                komsuluk_var = True
                        else:
                            if (tx > 0 and tahta[ty][tx-1] != '') or (tx < boyut-1 and tahta[ty][tx+1] != ''):
                                komsuluk_var = True
                            if i == 0 and ty > 0 and tahta[ty-1][tx] != '':
                                komsuluk_var = True
                            if i == kelime_uzunluk-1 and ty < boyut-1 and tahta[ty+1][tx] != '':
                                komsuluk_var = True

                if cakisma_var:
                    continue

                if cakisamayan_harfler == 0 and not komsuluk_var:
                    continue

                gecerli_pozisyonlar.append((kelime, x, y, yon))

    return gecerli_pozisyonlar
#################################################################

def kelime_yerlestir_ve_puanla4(kelime, x_koord, y_koord, orientation, board, tahta_puanlari2, sozluk):
    import copy
    board_old = copy.deepcopy(board)
    harf_puanlari = {
        'A': 1, 'B': 3, 'C': 4, 'Ç': 4, 'D': 3, 'E': 1, 'F': 7, 'G': 5, 'Ğ': 8,
        'H': 5, 'I': 2, 'İ': 1, 'J': 10, 'K': 1, 'L': 1, 'M': 2, 'N': 1, 'O': 2,
        'Ö': 7, 'P': 5, 'R': 1, 'S': 2, 'Ş': 4, 'T': 1, 'U': 2, 'Ü': 3, 'V': 7,
        'Y': 3, 'Z': 4
    }
    
    # Orijinal kelime ve uzunluğunu sakla
    orijinal_kelime = kelime
    orijinal_uzunluk = len(orijinal_kelime)

    # --- ORIENTATION NORMALİZASYONU ---
    ori = str(orientation).strip().lower()
    yatay_mi = ori.startswith("h")
    
    # --- Yerleşim Kontrolü ---
    kontrol_koordinatlar = []
    if yatay_mi:
        if x_koord + orijinal_uzunluk > 15:
            return 3  # sınır dışı
        for n in range(x_koord, x_koord + orijinal_uzunluk):
            kontrol_koordinatlar.append((n, y_koord - 1))
            kontrol_koordinatlar.append((n, y_koord + 1))
        kontrol_koordinatlar += [(x_koord - 1, y_koord), (x_koord + orijinal_uzunluk, y_koord)]
    else:
        if y_koord + orijinal_uzunluk > 15:
            return 3  # sınır dışı
        for n in range(y_koord, y_koord + orijinal_uzunluk):
            kontrol_koordinatlar.append((x_koord - 1, n))
            kontrol_koordinatlar.append((x_koord + 1, n))
        kontrol_koordinatlar += [(x_koord, y_koord - 1), (x_koord, y_koord + orijinal_uzunluk)]

    kontrol_koordinatlar = [(x, y) for x, y in kontrol_koordinatlar if 0 <= x < 15 and 0 <= y < 15]
    checkdigit = any(board[y][x] != "" for x, y in kontrol_koordinatlar)
    if not checkdigit and all(board[y_koord][x_koord + i] == "" if yatay_mi else board[y_koord + i][x_koord] == "" for i in range(orijinal_uzunluk)):
        return 1  # başka kelimeye değmiyor

    # --- Yerleştirme ve stok belirleme ---
    stoktan_dus = []
    stokindis = []
    x, y = x_koord, y_koord
    for harf in orijinal_kelime:
        if board[y][x] == "":
            stoktan_dus.append(harf)
            stokindis.append("0")
        elif _hucre_harf_eslesiyor_mu(board[y][x], harf):
            stokindis.append("1")
        else:
            return 2  # çakışma var
        if yatay_mi:
            x += 1
        else:
            y += 1

    # Eğer hiç harf düşmemişse (stoktan_dus boşsa) bu geçersiz bir hamledir
    if not stoktan_dus:
        return 4  # Hiç yeni harf eklenmemiş, aynı kelimeyi yazmaya çalışıyor

    # --- Harfleri geçici olarak board'a yerleştir ---
    x, y = x_koord, y_koord
    yeni_harf_sayac = 0
    joker_tam_kelime_indisleri = set()
    for i, harf in enumerate(orijinal_kelime):
        if stokindis[i] == "0":
            if yeni_harf_sayac in joker_stok_indisleri:
                board[y][x] = str(harf).lower()
                joker_tam_kelime_indisleri.add(i)
            else:
                board[y][x] = harf
            yeni_harf_sayac += 1
        if yatay_mi:
            x += 1
        else:
            y += 1

    # --- Kelimenin ÖNCESİNDEKİ harfleri bul ---
    on_ek = []
    if yatay_mi:
        x_bas = x_koord - 1
        while x_bas >= 0 and board[y_koord][x_bas] != "":
            on_ek.insert(0, _normalize_cell_letter(board[y_koord][x_bas]))
            stokindis.insert(0, "1")
            x_bas -= 1
    else:
        y_bas = y_koord - 1
        while y_bas >= 0 and board[y_bas][x_koord] != "":
            on_ek.insert(0, _normalize_cell_letter(board[y_bas][x_koord]))
            stokindis.insert(0, "1")
            y_bas -= 1

    # --- Kelimenin SONUNDAKİ harfleri bul ---
    son_ek = []
    if yatay_mi:
        x_end = x_koord + orijinal_uzunluk
        while x_end < 15 and board[y_koord][x_end] != "":
            son_ek.append(_normalize_cell_letter(board[y_koord][x_end]))
            stokindis.append("1")
            x_end += 1
    else:
        y_end = y_koord + orijinal_uzunluk
        while y_end < 15 and board[y_end][x_koord] != "":
            son_ek.append(_normalize_cell_letter(board[y_end][x_koord]))
            stokindis.append("1")
            y_end += 1

    # Tam kelimeyi oluştur
    tam_kelime = "".join(on_ek) + orijinal_kelime + "".join(son_ek)
    
    # Koordinatları güncelle
    if yatay_mi:
        x_koord_updated = x_koord - len(on_ek)
        y_koord_updated = y_koord
    else:
        x_koord_updated = x_koord
        y_koord_updated = y_koord - len(on_ek)

    # --- kelime_kontrol_final işlevini çağır ---
    from kelimelik_engine1 import kelime_kontrol_final
    olusan_kelimeler, puanlardizisi, toplam_puan, tahta_puanlari2 = kelime_kontrol_final(
        tam_kelime, x_koord_updated, y_koord_updated, orientation, board_old, harf_puanlari, tahta_puanlari2, stoktan_dus, stokindis
    )

    # Her kelime için sözlükte olup olmadığını kontrol et
    kelime_durumlari = [(k, k in sozluk) for k in olusan_kelimeler]

    # Tüm kelimeler sözlükte mi?
    olusan_kelimeler_sozlukte_var_mi = all(durum for (_, durum) in kelime_durumlari)

    
    return {
        "kelimeler": olusan_kelimeler,
        "kelime_durumlari": kelime_durumlari,
        "puan": toplam_puan,
        "oynanan_kelime_gosterim": _format_played_word_with_jokers(orijinal_kelime, stokindis[:len(orijinal_kelime)], joker_stok_indisleri),
        "board": board,
        "stoktan_dus": [(f"JOKER->{h}" if i in joker_stok_indisleri else h) for i, h in enumerate(stoktan_dus)],
        "yeni_tahta_puanlari": tahta_puanlari2,
        "gecerli": olusan_kelimeler_sozlukte_var_mi
    }




##############################################3

from itertools import permutations
from collections import Counter

def find_possible_words_and_orientations3(available_letters, word_parts, dictionary):
    """
    Girdilerin ve çıktının tamamen büyük harfli Türkçe karakterlerle olduğu versiyon.
    Aynı kelimenin tahtada zaten var olan haliyle tekrar üretilmesini engeller.
    """
    possible_words = set()
    available_counts = Counter(available_letters)  # Büyük harflerle sayım yapılır
    dictionary = {word for word in dictionary}     # Sözlük zaten büyük harfli olarak varsayılır

    for part, orientation in word_parts:
        max_extra_letters = len(available_letters)

        for pre_len in range(max_extra_letters + 1):
            for pre in permutations(available_letters, pre_len):
                pre_counter = Counter(pre)
                if not pre_counter <= available_counts:
                    continue

                remaining = list((available_counts - pre_counter).elements())
                for post_len in range(max_extra_letters - pre_len + 1):
                    for post in permutations(remaining, post_len):
                        post_counter = Counter(post)
                        total_counter = pre_counter + post_counter
                        if not total_counter <= available_counts:
                            continue

                        candidate = ''.join(pre) + part + ''.join(post)
                        if candidate != part and candidate in dictionary:
                            possible_words.add((candidate, orientation))

        # Sadece tek harfli parçalar doğrudan sözlükte varsa eklenir
        if len(part) == 1 and part in dictionary:
            possible_words.add((part, orientation))

    return sorted(possible_words)

#######################################################

from itertools import permutations
from collections import Counter

def find_possible_words_and_orientations4(available_letters, word_parts, dictionary, board_size=15):
    """
    Verilen kelimeciği tahtadaki konumunda sabit tutarak, başına ve/veya sonuna
    eldeki harfleri (JOKER dahil) ekleyerek sözlükteki geçerli kelimeleri üretir.

    Sadece tahtanın içinde kalan yerleşimler geçerli sayılır.
    """
    from collections import Counter

    possible_words = set()
    dictionary = [str(word).upper() for word in dictionary]
    rack_counter, joker_count = _parse_rack_and_jokers(available_letters)

    for part, (x_pos, y_pos), orientation in word_parts:
        part_u = str(part).upper()

        for candidate in dictionary:
            if candidate == part_u:
                continue

            idx = candidate.find(part_u)
            while idx != -1:
                pre = candidate[:idx]
                post = candidate[idx + len(part_u):]
                needed = pre + post

                if _kelime_can_be_formed_with_jokers(needed, rack_counter, joker_count):
                    if orientation == 'horizontal':
                        start_x = x_pos - len(pre)
                        start_y = y_pos
                        end_x = start_x + len(candidate) - 1
                        end_y = start_y
                    else:
                        start_x = x_pos
                        start_y = y_pos - len(pre)
                        end_x = start_x
                        end_y = start_y + len(candidate) - 1

                    if 0 <= start_x < board_size and 0 <= start_y < board_size and                        0 <= end_x < board_size and 0 <= end_y < board_size:
                        possible_words.add((candidate, (start_x, start_y), orientation))

                idx = candidate.find(part_u, idx + 1)

        if len(part_u) == 1 and part_u in dictionary:
            if 0 <= x_pos < board_size and 0 <= y_pos < board_size:
                possible_words.add((part_u, (x_pos, y_pos), orientation))

    return sorted(possible_words, key=lambda x: (x[2], x[0]))

#############################################

def aday_kelime_yerlestir2(tahta, aday_kelimeler):
    """
    Girdi: aday_kelimeler -> [('KELIME', 'horizontal'), ('KELIME2', 'vertical'), ...]
    Çıktı: [(kelime, (x, y), orientation)] - Belirtilen yöndeki geçerli yerleşim pozisyonları
    """
    boyut = len(tahta)
    gecerli_pozisyonlar = []

    for kelime, yon in aday_kelimeler:
        kelime_uzunluk = len(kelime)

        for y in range(boyut):
            for x in range(boyut):
                if (yon == 'horizontal' and x + kelime_uzunluk > boyut) or \
                   (yon == 'vertical' and y + kelime_uzunluk > boyut):
                    continue

                cakisma_var = False
                cakisamayan_harfler = 0
                komsuluk_var = False

                for i in range(kelime_uzunluk):
                    tx = x + (i if yon == 'horizontal' else 0)
                    ty = y + (i if yon == 'vertical' else 0)

                    if tahta[ty][tx] != '':
                        if not _hucre_harf_eslesiyor_mu(tahta[ty][tx], kelime[i]):
                            cakisma_var = True
                            break
                        cakisamayan_harfler += 1
                    else:
                        if yon == 'horizontal':
                            if (ty > 0 and tahta[ty-1][tx] != '') or (ty < boyut-1 and tahta[ty+1][tx] != ''):
                                komsuluk_var = True
                            if i == 0 and tx > 0 and tahta[ty][tx-1] != '':
                                komsuluk_var = True
                            if i == kelime_uzunluk-1 and tx < boyut-1 and tahta[ty][tx+1] != '':
                                komsuluk_var = True
                        else:
                            if (tx > 0 and tahta[ty][tx-1] != '') or (tx < boyut-1 and tahta[ty][tx+1] != ''):
                                komsuluk_var = True
                            if i == 0 and ty > 0 and tahta[ty-1][tx] != '':
                                komsuluk_var = True
                            if i == kelime_uzunluk-1 and ty < boyut-1 and tahta[ty+1][tx] != '':
                                komsuluk_var = True

                if cakisma_var:
                    continue

                if cakisamayan_harfler == 0 and not komsuluk_var:
                    continue

                gecerli_pozisyonlar.append((kelime, (x, y), yon))

    return gecerli_pozisyonlar


##########################################################################

def hesapla_dezavantaj_puani(board, yeni_harf_koordinatlari, tahta_puanlari2):
    """
    Her yeni yerleştirilen harfin tüm 8 yönündeki boş hücrelerdeki çarpanlara göre
    rakibe bırakılan potansiyel avantajı ölçer.
    """
    dezavantaj = 0
    for (x, y) in yeni_harf_koordinatlari:
        for dx, dy in [(-1,0), (-1,-1), (1,0), (0,-1), (0,1), (1,1), (1,-1), (-1,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 15 and 0 <= ny < 15 and board[ny][nx] == "":
                carpani = tahta_puanlari2[ny][nx]
                if carpani == 6:  # 3x Kelime
                    dezavantaj += 6
                elif carpani == 4:  # 2x Kelime
                    dezavantaj += 4
                elif carpani == 3:  # 3x Harf
                    dezavantaj += 3
                elif carpani == 2:  # 2x Harf
                    dezavantaj += 2
    return dezavantaj
##############################################################################
def kelime_yerlestir_ve_puanla5(kelime, x_koord, y_koord, orientation, board, tahta_puanlari2, sozluk, eldeki_harfler=None):
    import copy
    board_old = copy.deepcopy(board)
    harf_puanlari = {
        'A': 1, 'B': 3, 'C': 4, 'Ç': 4, 'D': 3, 'E': 1, 'F': 7, 'G': 5, 'Ğ': 8,
        'H': 5, 'I': 2, 'İ': 1, 'J': 10, 'K': 1, 'L': 1, 'M': 2, 'N': 1, 'O': 2,
        'Ö': 7, 'P': 5, 'R': 1, 'S': 2, 'Ş': 4, 'T': 1, 'U': 2, 'Ü': 3, 'V': 7,
        'Y': 3, 'Z': 4
    }

    orijinal_kelime = _normalize_input_word(kelime)
    orijinal_uzunluk = len(kelime)

    
    # --- ORIENTATION NORMALİZASYONU ---
    ori = str(orientation).strip().lower()
    yatay_mi = ori.startswith("h")   # 'h' veya 'horizontal' -> yata

    kontrol_koordinatlar = []
    #if yatay_mi:
    if yatay_mi:
        if x_koord + orijinal_uzunluk > 15:
            return 3
        for n in range(x_koord, x_koord + orijinal_uzunluk):
            kontrol_koordinatlar.append((n, y_koord - 1))
            kontrol_koordinatlar.append((n, y_koord + 1))
        kontrol_koordinatlar += [(x_koord - 1, y_koord), (x_koord + orijinal_uzunluk, y_koord)]
    else:
        if y_koord + orijinal_uzunluk > 15:
            return 3
        for n in range(y_koord, y_koord + orijinal_uzunluk):
            kontrol_koordinatlar.append((x_koord - 1, n))
            kontrol_koordinatlar.append((x_koord + 1, n))
        kontrol_koordinatlar += [(x_koord, y_koord - 1), (x_koord, y_koord + orijinal_uzunluk)]

    kontrol_koordinatlar = [(x, y) for x, y in kontrol_koordinatlar if 0 <= x < 15 and 0 <= y < 15]
    checkdigit = any(board[y][x] != "" for x, y in kontrol_koordinatlar)
    if not checkdigit and all(board[y_koord][x_koord + i] == "" if yatay_mi else board[y_koord + i][x_koord] == "" for i in range(orijinal_uzunluk)):
        return 1

    stoktan_dus = []
    stokindis = []
    x, y = x_koord, y_koord
    for harf in orijinal_kelime:
        if board[y][x] == "":
            stoktan_dus.append(harf)
            stokindis.append("0")
        elif _hucre_harf_eslesiyor_mu(board[y][x], harf):
            stokindis.append("1")
        else:
            return 2
        if yatay_mi:
            x += 1
        else:
            y += 1

    if not stoktan_dus:
        return 4

    joker_stok_indisleri = _joker_indislerini_bul(stoktan_dus, eldeki_harfler)

    x, y = x_koord, y_koord
    yeni_harf_sayac = 0
    joker_tam_kelime_indisleri = set()
    for i, harf in enumerate(orijinal_kelime):
        if stokindis[i] == "0":
            if yeni_harf_sayac in joker_stok_indisleri:
                board[y][x] = str(harf).lower()
                joker_tam_kelime_indisleri.add(i)
            else:
                board[y][x] = harf
            yeni_harf_sayac += 1
        if yatay_mi:
            x += 1
        else:
            y += 1

    on_ek = []
    if yatay_mi:
        x_bas = x_koord - 1
        while x_bas >= 0 and board[y_koord][x_bas] != "":
            on_ek.insert(0, _normalize_cell_letter(board[y_koord][x_bas]))
            stokindis.insert(0, "1")
            x_bas -= 1
    else:
        y_bas = y_koord - 1
        while y_bas >= 0 and board[y_bas][x_koord] != "":
            on_ek.insert(0, _normalize_cell_letter(board[y_bas][x_koord]))
            stokindis.insert(0, "1")
            y_bas -= 1

    son_ek = []
    if yatay_mi:
        x_end = x_koord + orijinal_uzunluk
        while x_end < 15 and board[y_koord][x_end] != "":
            son_ek.append(_normalize_cell_letter(board[y_koord][x_end]))
            stokindis.append("1")
            x_end += 1
    else:
        y_end = y_koord + orijinal_uzunluk
        while y_end < 15 and board[y_end][x_koord] != "":
            son_ek.append(_normalize_cell_letter(board[y_end][x_koord]))
            stokindis.append("1")
            y_end += 1

    tam_kelime = "".join(on_ek) + orijinal_kelime + "".join(son_ek)

    if yatay_mi:
        x_koord_updated = x_koord - len(on_ek)
        y_koord_updated = y_koord
    else:
        x_koord_updated = x_koord
        y_koord_updated = y_koord - len(on_ek)

    joker_tam_kelime_indisleri = {idx + len(on_ek) for idx in joker_tam_kelime_indisleri}

    olusan_kelimeler, puanlardizisi, toplam_puan, tahta_puanlari2 = kelime_kontrol_final(
        tam_kelime, x_koord_updated, y_koord_updated, orientation, board_old, harf_puanlari, tahta_puanlari2, stoktan_dus, stokindis, jokerindis=joker_tam_kelime_indisleri
    )

    if len(stoktan_dus) == 7:
        toplam_puan += 35

    kelime_durumlari = [(_normalize_cell_letter(k), _normalize_cell_letter(k) in sozluk) for k in olusan_kelimeler]
    olusan_kelimeler_sozlukte_var_mi = all(durum for (_, durum) in kelime_durumlari)

    # --- DEZAVANTAJ PUANI HESABI ---
    tek_harf_koordinatlari = []
    x, y = x_koord_updated, y_koord_updated
    for i, kel in enumerate(olusan_kelimeler):
        if len(kel) == 1 and stokindis[i] == "0":  # sadece tek harfli olanlar ve yeni yerleştirilenler
            tek_harf_koordinatlari.append((x, y))
        if yatay_mi:
            x += 1
        else:
            y += 1

    #dezavantaj_puani = hesapla_dezavantaj_puani(board, tek_harf_koordinatlari, tahta_puanlari2)
    dezavantaj_puani = round(hesapla_dezavantaj_puani_v2(board, tek_harf_koordinatlari, tahta_puanlari2),2)

    return {
        "kelimeler": olusan_kelimeler,
        "kelime_durumlari": kelime_durumlari,
        "puan": toplam_puan,
        "oynanan_kelime_gosterim": _format_played_word_with_jokers(orijinal_kelime, stokindis[:len(orijinal_kelime)], joker_stok_indisleri),
        "board": board,
        "stoktan_dus": [(f"JOKER->{h}" if i in joker_stok_indisleri else h) for i, h in enumerate(stoktan_dus)],
        "yeni_tahta_puanlari": tahta_puanlari2,
        "gecerli": olusan_kelimeler_sozlukte_var_mi,
        "dezavantaj": dezavantaj_puani
    }

########################################################

def harf_dagit(harf_stogu, n):
    import random
    elde_edilen = []
    mevcut_stok = harf_stogu.copy()

    tum_harfler = [harf for harf, adet in mevcut_stok.items() for _ in range(adet)]
    random.shuffle(tum_harfler)

    for harf in tum_harfler:
        if len(elde_edilen) >= n:
            break
        if mevcut_stok[harf] > 0:
            elde_edilen.append(harf)
            mevcut_stok[harf] -= 1

    return elde_edilen, mevcut_stok

#########################################################

def hamle(board , tahta_puanlari2 , eldeki_harfler , sozluk2):
    import kelimelik_engine1 as ke
    import copy
    #### TAHTADA VAR OLAN KELİMECİK GRUPLARI
    cikarilan=ke.extract_words(board)
    kelimecik=[i[0] for i in cikarilan]
    filtered_list_coklu = [item for item in cikarilan if len(item[0]) >= 2]
    filtered_list_tekli = [item for item in cikarilan if len(item[0]) == 1]
    coklu_kelimecik = [a[0] for a in filtered_list_coklu]
    tekli_kelimecik = [a[0] for a in filtered_list_tekli]
    #### KELİMECİKLERİDEN OLUŞMUŞ , SOZLUKTE KARŞILIĞI OLAN KELİMELER
    filtreli_sozluk2=ke.filter_scrabble_dictionary(sozluk2,coklu_kelimecik).tolist()
    filtreli_sozluk1=ke.filter_scrabble_dictionary(sozluk2,tekli_kelimecik).tolist()
    #eldeki_harfler=input("ELİNİZDEKİ HARFLERİ GİRİNİZ")
    #eldeki_harfler=input("Elinizdeki Harfler:")
    mumkun_kelimeler=ke.find_possible_words_and_orientations4(eldeki_harfler,filtered_list_coklu,filtreli_sozluk2)
    mumkun_kelimeler+=(ke.find_possible_words_and_orientations4(eldeki_harfler, filtered_list_tekli , filtreli_sozluk1))
    eldekinden_kelimeler=(ke.find_valid_words_from_available_with_orientations(eldeki_harfler,sozluk2))
    mumkun_kelimeler+=ke.aday_kelime_yerlestir2(board,eldekinden_kelimeler)
    print(len(mumkun_kelimeler))
    board_temp=copy.deepcopy(board)
    tahta_puanlari_temp=copy.deepcopy(tahta_puanlari2)
    kelime_gecerli = []
    xkoord_gecerli = []
    ykoord_gecerli = []
    orient_gecerli = []
    puan_gecerli = []
    dezavantaj_gecerli = []
    dezavantaj_basina_puan_gecerli=[]
    harf_basina_gecerli=[]
    kullanilan_max=[]
    kullanilan_ortmax=[]
    
    # Initialize temporary board and scores
    board_temp = copy.deepcopy(board)
    harf_basina=0
    max_puan=0
    max_kelime=""
    max_x_koord=0
    max_y_koord=0
    max_orient=""
    max_harfbasi_puan=0
    
    # Yeni bir dizi oluşturuyoruz (tahta_puanlari'nin bir kopyası)
    yeni_tahta_puanlari = tahta_puanlari2.copy()
    # Board'da boş olan hücrelerin indekslerini bulup, yeni_puanlar'da bu hücreleri 0 yapıyoruz
    yeni_tahta_puanlari[board != ''] = 0
    for a in range(len(mumkun_kelimeler)):
        kelime = mumkun_kelimeler[a][0]
        x_koord = mumkun_kelimeler[a][1][0]
        y_koord = mumkun_kelimeler[a][1][1]
        orient = mumkun_kelimeler[a][2][0]
        
        cikti = ke.kelime_yerlestir_ve_puanla5(
            kelime.upper(), 
            x_koord, 
            y_koord, 
            orient,
            board_temp,
            yeni_tahta_puanlari, 
            sozluk2,
            eldeki_harfler=eldeki_harfler
        )
        #harf_basina=int(cikti["puan"])/len(kelime)
        # Eğer çıktı dictionary değilse bu iterasyonu atla
        if not isinstance(cikti, dict):
            board_temp = copy.deepcopy(board)
            #tahta_puanlari_temp = copy.deepcopy(yeni_tahta_puanlari)
            continue  # Bir sonraki döngüye geç
        
        # Eğer dictionary ise işlemlere devam et
        if cikti["gecerli"]:
            harf_basina=int(cikti["puan"])/len(kelime)
            #harf_basina_gecerli.append("%.2f" %harf_basina)
            harf_basina_gecerli.append(round(harf_basina,2))
            puan_gecerli.append(cikti["puan"])
            kelime_gecerli.append(kelime)
            xkoord_gecerli.append(x_koord)
            ykoord_gecerli.append(y_koord)
            orient_gecerli.append(orient)
            dezavantaj_gecerli.append(cikti["dezavantaj"])
            dezavantaj_basina_puan_gecerli.append(round(cikti["dezavantaj"]/cikti["puan"],2))
            if cikti["puan"]>max_puan:
                max_puan=cikti["puan"]
                max_kelime=kelime
                max_x_koord=x_koord
                max_y_koord=y_koord
                max_orient=orient
                max_harf_basina=harf_basina
                max_dezavantaj=cikti["dezavantaj"]
                kullanilan_max=cikti["stoktan_dus"]
            if max_harfbasi_puan<harf_basina:
                max_harfbasi_puan=harf_basina
                max_harfbasi_kelime=kelime
                max_harfbasi_toplam_puan=cikti["puan"]
                max_harfbasi_x=x_koord
                max_harfbasi_y=y_koord
                max_harfbasi_orient=orient
                max_harfbasi_dezavantaj=cikti["dezavantaj"]
                kullanilan_ortmax=cikti["stoktan_dus"]
                
                
        # Reset the board only if the word was invalid
        #if not cikti["gecerli"]:
        board_temp = copy.deepcopy(board)
            #tahta_puanlari_temp = copy.deepcopy(yeni_tahta_puanlari)
        
    #ana_dizin=zip(kelime_gecerli,xkoord_gecerli,ykoord_gecerli,orient_gecerli,puan_gecerli,harf_basina_gecerli,dezavantaj_gecerli)
    ana_dizin=zip(kelime_gecerli,xkoord_gecerli,ykoord_gecerli,orient_gecerli,puan_gecerli,harf_basina_gecerli,dezavantaj_gecerli ,dezavantaj_basina_puan_gecerli)
    ana_dizin=sorted(ana_dizin, reverse=True , key=lambda tup: (tup[4],tup[5]) )
    #print("Geçerli Kelimeler:", set(kelime_gecerli))
    #print("Geçerli Kelimeler:", kelime_gecerli)
    #print("Geçerli Puanlar:", puan_gecerli)
    #print("Harf Basina Puanlar:", harf_basina_gecerli)
    #print(tuple(ana_dizin))
    #print("----------------------------")
    print( max_kelime , "kelimesi max puan alan kelimedir ve aldığı puan", max_puan)
    #print("Max_kelime:", max_kelime)
    #print(max_kelime , "kelimesinin x koordinatı:", max_x_koord)
    #print(max_kelime , "kelimesinin y koordinatı:", max_y_koord)
    #print(max_kelime , "kelimesinin dizilimi:", max_orient)
    #print(max_kelime , "kelimesinin harf basina puanı:", max_harf_basina)
    #print(max_kelime , "kelimesinin dezavantaj puanı:", max_dezavantaj)
    #print("----------------------------")
    #print(max_harfbasi_kelime , "kelimesi harf basina en yüksek puan alan kelimedir " , max_harfbasi_puan ," alır")
    #print(max_harfbasi_kelime , "kelimesinin aldığı toplam puan" , max_harfbasi_toplam_puan ,"dır")
    #print(max_harfbasi_kelime , "kelimesinin x koordinatı" , max_harfbasi_x ,"dır")
    #print(max_harfbasi_kelime , "kelimesinin y koordinatı" , max_harfbasi_y ,"dır")
    #print(max_harfbasi_kelime , "kelimesinin dizilimi" , max_harfbasi_orient ,"dır")
    #print(max_harfbasi_kelime , "dezavantaj puanı" , max_harfbasi_dezavantaj ,"dır")
    
        
    final_islem=ke.kelime_yerlestir_ve_puanla(
                max_kelime.upper(), 
                max_x_koord, 
                max_y_koord, 
                max_orient,
                board,
                yeni_tahta_puanlari, 
                sozluk2
        )
    #ke.print_board(board)
    return (board , kullanilan_max , max_puan,ana_dizin)


########################################################################################
def hesapla_dezavantaj_puani_v2(board, yeni_harf_koordinatlari, tahta_puanlari2):
    """
    Seviye-2 dezavantaj skoru:
      - Çarpan türü (TL/DL/DW/TW) taban değeri
      - Pozisyonel risk (uç/gövde farkı)
      - Erişilebilirlik (bağlantı var mı, boşluk uzunluğu, merkez yakınlığı ops.)
      - Yön uyumu (harf/kelime çarpanı için uygunluk)
      - Yakınlık zayıflaması (mesafe 1..3)
    Not: 'yeni_harf_koordinatlari' liste[(x,y)] — yeni yerleştirilen harfler.
    """
    import numpy as np

    if not yeni_harf_koordinatlari:
        return 0

    # --- Ayarlanabilir katsayılar ---
    BONUS_BASE = {  # tahta_puanlari2 değer haritası
        6: 6.0,   # 3x Kelime (TW)
        4: 4.0,   # 2x Kelime (DW)
        3: 6.0,   # 3x Harf   (TL)
        2: 3.0,   # 2x Harf   (DL)
        0: 0.0
    }
    POZ_UC   = 1.8
    POZ_GOVDE= 0.9

    ERIS_BAGLANTI_VAR = 1.2
    ERIS_BAGLANTI_YOK = 1.0
    # Boşluk uzunluğu normalizasyonu: min(1, uzunluk/7)
    MESAFE_ZAYIF = {1: 1.00, 2: 0.75, 3: 0.50}

    # Yön uyumu
    YON_UYUM_HARF_TEKE_YAKIN = 1.30   # DL/TL ve tek-harfle tetiklenebilir bağlantı varsa
    YON_UYUM_KELIME_HAT = 1.20        # DW/TW ve hat üzerinde en az 2 boşluk varsa
    YON_UYUM_YOK  = 1.00

    H, W = 15, 15
    board = np.array(board)  # güvenlik

    # --- Orientation ve uç/gövde heuristiği ---
    xs = [x for x, _ in yeni_harf_koordinatlari]
    ys = [y for _, y in yeni_harf_koordinatlari]
    horiz = len(set(ys)) == 1 and len(xs) >= 2
    vert  = len(set(xs)) == 1 and len(ys) >= 2

    uc_set = set()
    if horiz:
        y0 = ys[0]
        x_min, x_max = min(xs), max(xs)
        uc_set.add((x_min, y0)); uc_set.add((x_max, y0))
    elif vert:
        x0 = xs[0]
        y_min, y_max = min(ys), max(ys)
        uc_set.add((x0, y_min)); uc_set.add((x0, y_max))
    else:
        # tek harf ya da dağınık — hepsini uç gibi davran
        uc_set = set(yeni_harf_koordinatlari)

    def bos_mu(x, y):
        return (0 <= x < W and 0 <= y < H and board[y, x] == "")

    def dolu_mu(x, y):
        return (0 <= x < W and 0 <= y < H and board[y, x] != "")

    def bonus_degeri(x, y):
        v = tahta_puanlari2[y][x]
        return BONUS_BASE.get(int(v), 0.0)

    def baglanti_var_mi(x, y):
        # Ortogonal komşularda mevcut harf var mı?
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx, ny = x+dx, y+dy
            if dolu_mu(nx, ny):
                return True
        return False

    def bosluk_uzunlugu(x, y, dx, dy, limit=7):
        # (x,y) dahil tek yönde kesintisiz boşluk sayısı
        cnt = 1
        ix, iy = x+dx, y+dy
        while 0 <= ix < W and 0 <= iy < H and board[iy, ix] == "" and cnt < limit:
            cnt += 1
            ix += dx; iy += dy
        return cnt

    def longest_run_norm(x, y):
        # Yatay ve dikeyde en uzun boş hat (iki yön birleşik), 1..7 normalize
        left  = bosluk_uzunlugu(x, y, -1, 0)
        right = bosluk_uzunlugu(x, y,  1, 0)
        up    = bosluk_uzunlugu(x, y,  0,-1)
        down  = bosluk_uzunlugu(x, y,  0, 1)
        horz_total = min(7, left + right - 1)
        vert_total = min(7, up + down - 1)
        return min(1.0, max(horz_total, vert_total) / 7.0), horz_total, vert_total

    def yon_uyumu_katsayi(x, y, bonus_raw):
        # Harf çarpanı: tek harf bağlantı ile kolay tetikleniyorsa artır
        # Kelime çarpanı: aynı hat üzerinde en az 2 uzunluk boşluk varsa artır
        if bonus_raw in (3, 2):  # TL/DL
            return YON_UYUM_HARF_TEKE_YAKIN if baglanti_var_mi(x, y) else YON_UYUM_YOK
        elif bonus_raw in (6, 4):  # TW/DW
            norm, horz_len, vert_len = longest_run_norm(x, y)
            if max(horz_len, vert_len) >= 2:
                return YON_UYUM_KELIME_HAT
            return YON_UYUM_YOK
        return YON_UYUM_YOK

    toplam = 0.0

    # Mesafe 1..3 içinde boş çarpan hücreleri tara
    for (sx, sy) in yeni_harf_koordinatlari:
        poz_katsayi = POZ_UC if (sx, sy) in uc_set else POZ_GOVDE

        for radius in (1, 2, 3):
            att = MESAFE_ZAYIF[radius]
            for dx in range(-radius, radius+1):
                for dy in range(-radius, radius+1):
                    if max(abs(dx), abs(dy)) != radius:
                        continue  # sadece dış halka
                    nx, ny = sx+dx, sy+dy
                    if not (0 <= nx < W and 0 <= ny < H):
                        continue
                    if not bos_mu(nx, ny):
                        continue

                    raw = int(tahta_puanlari2[ny][nx])
                    base = BONUS_BASE.get(raw, 0.0)
                    if base <= 0.0:
                        continue

                    eris = ERIS_BAGLANTI_VAR if baglanti_var_mi(nx, ny) else ERIS_BAGLANTI_YOK
                    norm, _, _ = longest_run_norm(nx, ny)
                    yonk = yon_uyumu_katsayi(nx, ny, raw)

                    skor = base * poz_katsayi * eris * norm * yonk * att
                    toplam += skor

    return toplam

##########################################################################
'''
def hamle_dezavantaj_oran(board , tahta_puanlari2 , eldeki_harfler , sozluk2):
    import kelimelik_engine1 as ke
    import copy
    
    #### TAHTADA VAR OLAN KELİMECİK GRUPLARI
    cikarilan=ke.extract_words(board)
    filtered_list_coklu = [item for item in cikarilan if len(item[0]) >= 2]
    filtered_list_tekli = [item for item in cikarilan if len(item[0]) == 1]
    coklu_kelimecik = [a[0] for a in filtered_list_coklu]
    tekli_kelimecik = [a[0] for a in filtered_list_tekli]

    #### KELİMECİKLERDEN OLUŞMUŞ , SÖZLÜKTE KARŞILIĞI OLAN KELİMELER
    filtreli_sozluk2=ke.filter_scrabble_dictionary(sozluk2,coklu_kelimecik).tolist()
    filtreli_sozluk1=ke.filter_scrabble_dictionary(sozluk2,tekli_kelimecik).tolist()

    mumkun_kelimeler=ke.find_possible_words_and_orientations4(eldeki_harfler,filtered_list_coklu,filtreli_sozluk2)
    mumkun_kelimeler+=(ke.find_possible_words_and_orientations4(eldeki_harfler, filtered_list_tekli , filtreli_sozluk1))
    eldekinden_kelimeler=(ke.find_valid_words_from_available_with_orientations(eldeki_harfler,sozluk2))
    mumkun_kelimeler+=ke.aday_kelime_yerlestir2(board,eldekinden_kelimeler)

    board_temp=copy.deepcopy(board)
    
    kelime_gecerli = []
    xkoord_gecerli = []
    ykoord_gecerli = []
    orient_gecerli = []
    puan_gecerli = []
    dezavantaj_gecerli = []
    dezavantaj_basina_puan_gecerli=[]
    harf_basina_gecerli=[]
    kullanilan_max=[]
    
    min_oran = float("inf")
    secilen_kelime=""
    secilen_x=0
    secilen_y=0
    secilen_orient=""
    secilen_puan=0
    
    yeni_tahta_puanlari = tahta_puanlari2.copy()
    yeni_tahta_puanlari[board != ''] = 0

    for kelime, (x_koord, y_koord), orient in mumkun_kelimeler:
        cikti = ke.kelime_yerlestir_ve_puanla5(
            kelime.upper(), 
            x_koord, 
            y_koord, 
            orient,
            board_temp,
            yeni_tahta_puanlari, 
            sozluk2,
            eldeki_harfler=eldeki_harfler
        )
        if not isinstance(cikti, dict):
            board_temp = copy.deepcopy(board)
            continue  
        
        if cikti["gecerli"]:
            puan = cikti["puan"]
            dezavantaj = cikti["dezavantaj"]
            oran = round(dezavantaj/puan, 4) if puan>0 else float("inf")
            
            # listeye kaydet
            kelime_gecerli.append(kelime)
            xkoord_gecerli.append(x_koord)
            ykoord_gecerli.append(y_koord)
            orient_gecerli.append(orient)
            puan_gecerli.append(puan)
            dezavantaj_gecerli.append(dezavantaj)
            dezavantaj_basina_puan_gecerli.append(oran)
            harf_basina_gecerli.append(round(puan/len(kelime),2))
            
            # --- seçim: oran küçük olanı al ---
            if oran < min_oran:
                min_oran = oran
                secilen_kelime = kelime
                secilen_x = x_koord
                secilen_y = y_koord
                secilen_orient = orient
                secilen_puan = puan
                kullanilan_max = cikti["stoktan_dus"]

        board_temp = copy.deepcopy(board)
        
    ana_dizin = list(zip(
        kelime_gecerli, xkoord_gecerli, ykoord_gecerli, orient_gecerli,
        puan_gecerli, harf_basina_gecerli, dezavantaj_gecerli, dezavantaj_basina_puan_gecerli
    ))
    # küçükten büyüğe sıralama (en güvenli başta)
    ana_dizin = sorted(ana_dizin, key=lambda tup: tup[7])
    
    print(secilen_kelime , "kelimesi en düşük dezavantaj/puan oranına sahip. Oran:", min_oran)
    
    final_islem=ke.kelime_yerlestir_ve_puanla(
                secilen_kelime.upper(), 
                secilen_x, 
                secilen_y, 
                secilen_orient,
                board,
                yeni_tahta_puanlari, 
                sozluk2
        )
    return (board , kullanilan_max , secilen_puan, ana_dizin)
'''
########################################################################

def hamle_dezavantaj_oran(board , tahta_puanlari2 , eldeki_harfler , sozluk2):
    import kelimelik_engine1 as ke
    import copy
    
    #### TAHTADA VAR OLAN KELİMECİK GRUPLARI
    cikarilan = ke.extract_words(board)
    filtered_list_coklu = [item for item in cikarilan if len(item[0]) >= 2]
    filtered_list_tekli = [item for item in cikarilan if len(item[0]) == 1]
    coklu_kelimecik = [a[0] for a in filtered_list_coklu]
    tekli_kelimecik = [a[0] for a in filtered_list_tekli]

    #### KELİMECİKLERDEN OLUŞMUŞ , SÖZLÜKTE KARŞILIĞI OLAN KELİMELER
    filtreli_sozluk2 = ke.filter_scrabble_dictionary(sozluk2, coklu_kelimecik).tolist()
    filtreli_sozluk1 = ke.filter_scrabble_dictionary(sozluk2, tekli_kelimecik).tolist()

    mumkun_kelimeler = ke.find_possible_words_and_orientations4(eldeki_harfler, filtered_list_coklu, filtreli_sozluk2)
    mumkun_kelimeler += ke.find_possible_words_and_orientations4(eldeki_harfler, filtered_list_tekli, filtreli_sozluk1)
    eldekinden_kelimeler = ke.find_valid_words_from_available_with_orientations(eldeki_harfler, sozluk2)
    mumkun_kelimeler += ke.aday_kelime_yerlestir2(board, eldekinden_kelimeler)

    board_temp = copy.deepcopy(board)
    
    kelime_gecerli = []
    xkoord_gecerli = []
    ykoord_gecerli = []
    orient_gecerli = []
    puan_gecerli = []
    dezavantaj_gecerli = []
    dezavantaj_basina_puan_gecerli = []
    harf_basina_gecerli = []
    kullanilan_max = []
    
    min_oran = float("inf")
    secilen_kelime = ""
    secilen_x = 0
    secilen_y = 0
    secilen_orient = ""
    secilen_puan = 0
    
    yeni_tahta_puanlari = tahta_puanlari2.copy()
    yeni_tahta_puanlari[board != ''] = 0

    for kelime, (x_koord, y_koord), orient in mumkun_kelimeler:
        cikti = ke.kelime_yerlestir_ve_puanla5(
            kelime.upper(), 
            x_koord, 
            y_koord, 
            orient,
            board_temp,
            yeni_tahta_puanlari, 
            sozluk2,
            eldeki_harfler=eldeki_harfler
        )
        if not isinstance(cikti, dict):
            board_temp = copy.deepcopy(board)
            continue  
        
        if cikti["gecerli"]:
            puan = cikti["puan"]
            dezavantaj = cikti["dezavantaj"]
            oran = round(dezavantaj/puan, 4) if puan > 0 else float("inf")
            
            # listeye kaydet
            kelime_gecerli.append(kelime)
            xkoord_gecerli.append(x_koord)
            ykoord_gecerli.append(y_koord)
            orient_gecerli.append(orient)
            puan_gecerli.append(puan)
            dezavantaj_gecerli.append(dezavantaj)
            dezavantaj_basina_puan_gecerli.append(oran)
            harf_basina_gecerli.append(round(puan/len(kelime), 2))
            
            # --- seçim: oran küçük olanı al, eşitse puanı yüksek olanı tercih et ---
            if oran < min_oran or (oran == min_oran and puan > secilen_puan):
                min_oran = oran
                secilen_kelime = kelime
                secilen_x = x_koord
                secilen_y = y_koord
                secilen_orient = orient
                secilen_puan = puan
                kullanilan_max = cikti["stoktan_dus"]

        board_temp = copy.deepcopy(board)
        
    ana_dizin = list(zip(
        kelime_gecerli, xkoord_gecerli, ykoord_gecerli, orient_gecerli,
        puan_gecerli, harf_basina_gecerli, dezavantaj_gecerli, dezavantaj_basina_puan_gecerli
    ))
    # küçükten büyüğe sıralama (en güvenli başta, eşit oranlarda puan sırasına göre)
    ana_dizin = sorted(ana_dizin, key=lambda tup: (tup[7], -tup[4]))
    
    print(secilen_kelime , "kelimesi seçildi. Oran:", min_oran, " Puan:", secilen_puan)
    
    final_islem = ke.kelime_yerlestir_ve_puanla(
        secilen_kelime.upper(), 
        secilen_x, 
        secilen_y, 
        secilen_orient,
        board,
        yeni_tahta_puanlari, 
        sozluk2
    )
    return (board , kullanilan_max , secilen_puan, ana_dizin)

#################################################################
'''
def hamle_cok_kriterli(board, tahta_puanlari2, eldeki_harfler, sozluk2,
                        w_puan=0.45, w_harf=0.20, w_dez=0.20, w_oran=0.15):
    import kelimelik_engine1 as ke
    import copy, math

    def _as_float(x):
        try:
            return float(x)
        except:
            try:
                return float(str(x).replace(",", "."))
            except:
                return 0.0

    def _minmax(vals):
        if not vals:
            return []
        lo, hi = min(vals), max(vals)
        if math.isclose(lo, hi):
            return [0.0] * len(vals)
        rng = hi - lo
        return [(v - lo) / rng for v in vals]

    # 1) Tahtadan parçalar ve sözlük daraltma
    cikarilan = ke.extract_words(board)
    filtered_list_coklu = [item for item in cikarilan if len(item[0]) >= 2]
    filtered_list_tekli = [item for item in cikarilan if len(item[0]) == 1]
    coklu_kelimecik = [a[0] for a in filtered_list_coklu]
    tekli_kelimecik = [a[0] for a in filtered_list_tekli]

    filtreli_sozluk2 = ke.filter_scrabble_dictionary(sozluk2, coklu_kelimecik).tolist()
    filtreli_sozluk1 = ke.filter_scrabble_dictionary(sozluk2, tekli_kelimecik).tolist()

    mumkun_kelimeler  = ke.find_possible_words_and_orientations4(eldeki_harfler, filtered_list_coklu, filtreli_sozluk2)
    mumkun_kelimeler += ke.find_possible_words_and_orientations4(eldeki_harfler, filtered_list_tekli, filtreli_sozluk1)
    eldekinden_kelimeler = ke.find_valid_words_from_available_with_orientations(eldeki_harfler, sozluk2)
    mumkun_kelimeler += ke.aday_kelime_yerlestir2(board, eldekinden_kelimeler)

    # 2) Simülasyon: geçerli hamleleri topla
    board_temp = copy.deepcopy(board)
    yeni_tahta_puanlari = tahta_puanlari2.copy()
    yeni_tahta_puanlari[board != ''] = 0

    kelime_gecerli, xkoord_gecerli, ykoord_gecerli, orient_gecerli = [], [], [], []
    puan_gecerli, harf_basina_gecerli, dezavantaj_gecerli, oran_gecerli = [], [], [], []
    kullanilan_max = []

    for kelime, (x_koord, y_koord), orient in mumkun_kelimeler:
        cikti = ke.kelime_yerlestir_ve_puanla5(
            kelime.upper(), x_koord, y_koord, orient,
            board_temp, yeni_tahta_puanlari, sozluk2,
            eldeki_harfler=eldeki_harfler
        )
        if not isinstance(cikti, dict):
            board_temp = copy.deepcopy(board)
            continue
        if cikti["gecerli"]:
            puan = float(cikti["puan"])
            dez  = float(cikti["dezavantaj"])
            hb   = puan / max(len(kelime), 1)
            oran = (dez / puan) if puan > 0 else float("inf")

            kelime_gecerli.append(kelime)
            xkoord_gecerli.append(x_koord)
            ykoord_gecerli.append(y_koord)
            orient_gecerli.append(orient)
            puan_gecerli.append(puan)
            harf_basina_gecerli.append(round(hb, 2))
            dezavantaj_gecerli.append(dez)
            oran_gecerli.append(oran)
        board_temp = copy.deepcopy(board)

    # Hiç hamle yoksa pas
    if not kelime_gecerli:
        return (board, [], 0, [])

    # 3) Kompozit skor: min-max normalize + ağırlıklar
    Pn = _minmax(puan_gecerli)
    Hn = _minmax([_as_float(x) for x in harf_basina_gecerli])
    Dn = _minmax(dezavantaj_gecerli)   # risk -> eksi
    Rn = _minmax(oran_gecerli)         # risk -> eksi

    skor = [w_puan*Pn[i] + w_harf*Hn[i] - w_dez*Dn[i] - w_oran*Rn[i] for i in range(len(kelime_gecerli))]

    # 4) En iyiyi seç (eşitlikte puan, sonra harf_bası ile kır)
    best_idx = max(
        range(len(kelime_gecerli)),
        key=lambda i: (skor[i], puan_gecerli[i], _as_float(harf_basina_gecerli[i]))
    )

    secilen_kelime = kelime_gecerli[best_idx]
    secilen_x      = xkoord_gecerli[best_idx]
    secilen_y      = ykoord_gecerli[best_idx]
    secilen_orient = orient_gecerli[best_idx]
    secilen_puan   = int(round(puan_gecerli[best_idx]))

    # 5) Tahtaya yerleştir
    _ = ke.kelime_yerlestir_ve_puanla(
        secilen_kelime.upper(), secilen_x, secilen_y, secilen_orient,
        board, yeni_tahta_puanlari, sozluk2
    )

    # 6) ana_dizin'i kompozit skora göre sıralayıp döndür (yapıyı bozmadan)
    ana_dizin = list(zip(
        kelime_gecerli, xkoord_gecerli, ykoord_gecerli, orient_gecerli,
        [int(round(p)) for p in puan_gecerli],
        harf_basina_gecerli,
        dezavantaj_gecerli,
        oran_gecerli
    ))
    # sıralama: önce kompozit skor, sonra puan, sonra harf_bası
    order = sorted(range(len(ana_dizin)),
                   key=lambda i: (skor[i], puan_gecerli[i], _as_float(harf_basina_gecerli[i])),
                   reverse=True)
    ana_dizin = [ana_dizin[i] for i in order]

    # Kullanılan harfleri bulmak için: seçilen simülasyon çıktısındaki 'stoktan_dus' değerine ihtiyaç var.
    # Tekrar hesapla (sadece seçilen için) ki kullanilan_max'ı döndürebilelim:
    tmp = ke.kelime_yerlestir_ve_puanla5(
        secilen_kelime.upper(), secilen_x, secilen_y, secilen_orient,
        copy.deepcopy(board), yeni_tahta_puanlari, sozluk2,
        eldeki_harfler=eldeki_harfler
    )
    if isinstance(tmp, dict) and "stoktan_dus" in tmp:
        kullanilan_max = tmp["stoktan_dus"]
    else:
        kullanilan_max = []

    print(f"{secilen_kelime} seçildi | puan={secilen_puan} | "
          f"hb={harf_basina_gecerli[best_idx]} | dez={dezavantaj_gecerli[best_idx]} | oran={oran_gecerli[best_idx]:.4f}")

    return (board, kullanilan_max, secilen_puan, ana_dizin)
'''
##########################################3
'''

def hamle_cok_kriterli(board, tahta_puanlari2, eldeki_harfler, sozluk2,
                        w_puan=0.45, w_harf=0.20, w_dez=0.20, w_oran=0.15):
    import copy, math
    import kelimelik_engine1 as ke

    def _as_float(x):
        try:
            return float(x)
        except:
            try:
                return float(str(x).replace(",", "."))
            except:
                return 0.0

    def _minmax(vals):
        if not vals:
            return []
        lo, hi = min(vals), max(vals)
        if math.isclose(lo, hi):
            return [0.0] * len(vals)
        rng = hi - lo
        return [(v - lo) / rng for v in vals]

    # 1) Tahtadan parçalar + sözlük daraltma
    cikarilan = ke.extract_words(board)
    filtered_list_coklu = [item for item in cikarilan if len(item[0]) >= 2]
    filtered_list_tekli = [item for item in cikarilan if len(item[0]) == 1]
    coklu_kelimecik = [a[0] for a in filtered_list_coklu]
    tekli_kelimecik = [a[0] for a in filtered_list_tekli]

    filtreli_sozluk2 = ke.filter_scrabble_dictionary(sozluk2, coklu_kelimecik).tolist()
    filtreli_sozluk1 = ke.filter_scrabble_dictionary(sozluk2, tekli_kelimecik).tolist()

    mumkun_kelimeler  = ke.find_possible_words_and_orientations4(eldeki_harfler, filtered_list_coklu, filtreli_sozluk2)
    mumkun_kelimeler += ke.find_possible_words_and_orientations4(eldeki_harfler, filtered_list_tekli, filtreli_sozluk1)
    eldekinden_kelimeler = ke.find_valid_words_from_available_with_orientations(eldeki_harfler, sozluk2)
    mumkun_kelimeler += ke.aday_kelime_yerlestir2(board, eldekinden_kelimeler)

    # 2) Simülasyon: geçerli adaylar
    board_temp = copy.deepcopy(board)
    yeni_tahta_puanlari = tahta_puanlari2.copy()
    yeni_tahta_puanlari[board != ''] = 0

    kelime_gecerli, xkoord_gecerli, ykoord_gecerli, orient_gecerli = [], [], [], []
    puan_gecerli, harf_basina_gecerli, dezavantaj_gecerli, oran_gecerli = [], [], [], []
    stoktan_dus_gecerli = []

    for kelime, (x_koord, y_koord), orient in mumkun_kelimeler:
        cikti = ke.kelime_yerlestir_ve_puanla5(
            kelime.upper(), x_koord, y_koord, orient,
            board_temp, yeni_tahta_puanlari, sozluk2,
            eldeki_harfler=eldeki_harfler
        )
        if not isinstance(cikti, dict):
            board_temp = copy.deepcopy(board); continue
        if cikti.get("gecerli"):
            puan = float(cikti["puan"])
            dez  = float(cikti["dezavantaj"])
            hb   = puan / max(len(kelime), 1)
            oran = (dez / puan) if puan > 0 else float("inf")

            kelime_gecerli.append(kelime)
            xkoord_gecerli.append(x_koord)
            ykoord_gecerli.append(y_koord)
            orient_gecerli.append(orient)
            puan_gecerli.append(puan)
            harf_basina_gecerli.append(round(hb, 2))
            dezavantaj_gecerli.append(dez)
            oran_gecerli.append(oran)
            # >>> stoktan_dus’u aynen topla
            stoktan_dus_gecerli.append(cikti.get("stoktan_dus", []))
        board_temp = copy.deepcopy(board)

    # Hamle yoksa pas
    if not kelime_gecerli:
        return (board, [], 0, [])

    # 3) Kompozit skor
    Pn = _minmax(puan_gecerli)
    Hn = _minmax([_as_float(x) for x in harf_basina_gecerli])
    Dn = _minmax(dezavantaj_gecerli)
    Rn = _minmax(oran_gecerli)
    skor = [w_puan*Pn[i] + w_harf*Hn[i] - w_dez*Dn[i] - w_oran*Rn[i] for i in range(len(kelime_gecerli))]

    # 4) En iyiyi seç (eşitlikte puan, sonra harf_bası)
    best_idx = max(
        range(len(kelime_gecerli)),
        key=lambda i: (skor[i], puan_gecerli[i], _as_float(harf_basina_gecerli[i]))
    )

    secilen_kelime = kelime_gecerli[best_idx]
    secilen_x      = xkoord_gecerli[best_idx]
    secilen_y      = ykoord_gecerli[best_idx]
    secilen_orient = orient_gecerli[best_idx]
    secilen_puan   = int(round(puan_gecerli[best_idx]))
    # >>> Python listesi olarak garanti et
    km = stoktan_dus_gecerli[best_idx]
    if isinstance(km, str):
        kullanilan_max = list(km)
    elif hasattr(km, "tolist"):
        kullanilan_max = list(km.tolist())
    else:
        try:
            kullanilan_max = list(km)
        except:
            kullanilan_max = []

    # 5) Seçimi tahtaya uygula
    _ = ke.kelime_yerlestir_ve_puanla(
        secilen_kelime.upper(), secilen_x, secilen_y, secilen_orient,
        board, yeni_tahta_puanlari, sozluk2
    )

    # 6) ana_dizin’i sırala ve döndür
    ana_dizin = list(zip(
        kelime_gecerli, xkoord_gecerli, ykoord_gecerli, orient_gecerli,
        [int(round(p)) for p in puan_gecerli],
        harf_basina_gecerli, dezavantaj_gecerli, oran_gecerli
    ))
    order = sorted(range(len(ana_dizin)),
                   key=lambda i: (skor[i], puan_gecerli[i], _as_float(harf_basina_gecerli[i])),
                   reverse=True)
    ana_dizin = [ana_dizin[i] for i in order]

    print(f"{secilen_kelime} seçildi | puan={secilen_puan} | "
          f"hb={harf_basina_gecerli[best_idx]} | dez={dezavantaj_gecerli[best_idx]} | oran={oran_gecerli[best_idx]:.4f}")

    return (board, kullanilan_max, secilen_puan, ana_dizin)
'''
#############################################################
'''
def hamle_cok_kriterli(board, tahta_puanlari2, eldeki_harfler, sozluk2,
                        w_puan=0.45, w_harf=0.20, w_dez=0.20, w_oran=0.15):
    """
    Çok-kriterli ajan:
      - ana_dizin'i oluşturur
      - min-max normalize edip +puan +harf_basi −dez −oran kompozit skor hesaplar
      - en iyi hamleyi TAHTAYA UYGULAR ve (board, kullanilan_max, secilen_puan, ana_dizin) döner

    ÖNEMLİ: 'kullanilan_max' bilgisini 'stoktan_dus' yerine TAHTADAN türeterek hesaplıyoruz.
    Böylece O2'nin 'kullandı/çekti' logları garanti doğru olur.
    """
    import copy, math
    import kelimelik_engine1 as ke

    def _as_float(x):
        try:
            return float(x)
        except:
            try:
                return float(str(x).replace(",", "."))
            except:
                return 0.0

    def _minmax(vals):
        if not vals:
            return []
        lo, hi = min(vals), max(vals)
        if math.isclose(lo, hi):
            return [0.0] * len(vals)
        rng = hi - lo
        return [(v - lo) / rng for v in vals]

    # 1) Tahtadaki parçalar ve sözlük daraltma
    cikarilan = ke.extract_words(board)
    filtered_list_coklu = [item for item in cikarilan if len(item[0]) >= 2]
    filtered_list_tekli = [item for item in cikarilan if len(item[0]) == 1]
    coklu_kelimecik = [a[0] for a in filtered_list_coklu]
    tekli_kelimecik = [a[0] for a in filtered_list_tekli]

    filtreli_sozluk2 = ke.filter_scrabble_dictionary(sozluk2, coklu_kelimecik).tolist()
    filtreli_sozluk1 = ke.filter_scrabble_dictionary(sozluk2, tekli_kelimecik).tolist()

    mumkun_kelimeler  = ke.find_possible_words_and_orientations4(eldeki_harfler, filtered_list_coklu, filtreli_sozluk2)
    mumkun_kelimeler += ke.find_possible_words_and_orientations4(eldeki_harfler, filtered_list_tekli, filtreli_sozluk1)
    eldekinden_kelimeler = ke.find_valid_words_from_available_with_orientations(eldeki_harfler, sozluk2)
    mumkun_kelimeler += ke.aday_kelime_yerlestir2(board, eldekinden_kelimeler)

    # 2) Adayları simüle et
    board_temp = copy.deepcopy(board)
    yeni_tahta_puanlari = tahta_puanlari2.copy()
    yeni_tahta_puanlari[board != ''] = 0

    kelime_gecerli, xkoord_gecerli, ykoord_gecerli, orient_gecerli = [], [], [], []
    puan_gecerli, harf_basina_gecerli, dezavantaj_gecerli, oran_gecerli = [], [], [], []

    for kelime, (x_koord, y_koord), orient in mumkun_kelimeler:
        cikti = ke.kelime_yerlestir_ve_puanla5(
            kelime.upper(), x_koord, y_koord, orient,
            board_temp, yeni_tahta_puanlari, sozluk2,
            eldeki_harfler=eldeki_harfler
        )
        if not isinstance(cikti, dict):
            board_temp = copy.deepcopy(board)
            continue
        if cikti.get("gecerli"):
            puan = float(cikti["puan"])
            dez  = float(cikti["dezavantaj"])
            hb   = puan / max(len(kelime), 1)
            oran = (dez / puan) if puan > 0 else float("inf")

            kelime_gecerli.append(kelime)
            xkoord_gecerli.append(x_koord)
            ykoord_gecerli.append(y_koord)
            orient_gecerli.append(orient)
            puan_gecerli.append(puan)
            harf_basina_gecerli.append(round(hb, 2))
            dezavantaj_gecerli.append(dez)
            oran_gecerli.append(oran)

        board_temp = copy.deepcopy(board)

    # Hamle yoksa pas
    if not kelime_gecerli:
        return (board, [], 0, [])

    # 3) Kompozit skor ( +puan +harf_basi −dez −oran )
    Pn = _minmax(puan_gecerli)
    Hn = _minmax([_as_float(x) for x in harf_basina_gecerli])
    Dn = _minmax(dezavantaj_gecerli)
    Rn = _minmax(oran_gecerli)
    skor = [w_puan*Pn[i] + w_harf*Hn[i] - w_dez*Dn[i] - w_oran*Rn[i] for i in range(len(kelime_gecerli))]

    # 4) En iyiyi seç (eşitlikte puan, sonra harf_bası)
    best_idx = max(
        range(len(kelime_gecerli)),
        key=lambda i: (skor[i], puan_gecerli[i], _as_float(harf_basina_gecerli[i]))
    )

    secilen_kelime = kelime_gecerli[best_idx]
    secilen_x      = xkoord_gecerli[best_idx]
    secilen_y      = ykoord_gecerli[best_idx]
    secilen_orient = orient_gecerli[best_idx]
    secilen_puan   = int(round(puan_gecerli[best_idx]))

    # --- (YENİ) KULLANILAN HARFLERİ TAHTADAN TÜRET ---
    def _used_from_board(board_arr, word, x, y, orient):
        used = []
        W = word.upper()
        if str(orient).lower().startswith("h"):  # yatay
            for i, ch in enumerate(W):
                if board_arr[x, y+i] == '' or board_arr[x, y+i] is None:
                    used.append(ch)
        else:  # düşey varsayıyoruz
            for i, ch in enumerate(W):
                if board_arr[x+i, y] == '' or board_arr[x+i, y] is None:
                    used.append(ch)
        return used

    kullanilan_max = _used_from_board(board, secilen_kelime, secilen_x, secilen_y, secilen_orient)

    # 5) Seçimi tahtaya uygula
    _ = ke.kelime_yerlestir_ve_puanla(
        secilen_kelime.upper(), secilen_x, secilen_y, secilen_orient,
        board, yeni_tahta_puanlari, sozluk2
    )

    # 6) ana_dizin’i kompozit skora göre sırala ve döndür
    ana_dizin = list(zip(
        kelime_gecerli, xkoord_gecerli, ykoord_gecerli, orient_gecerli,
        [int(round(p)) for p in puan_gecerli],
        harf_basina_gecerli, dezavantaj_gecerli, oran_gecerli
    ))
    order = sorted(range(len(ana_dizin)),
                   key=lambda i: (skor[i], puan_gecerli[i], _as_float(harf_basina_gecerli[i])),
                   reverse=True)
    ana_dizin = [ana_dizin[i] for i in order]

    print(f"{secilen_kelime} seçildi | puan={secilen_puan} | "
          f"hb={harf_basina_gecerli[best_idx]} | dez={dezavantaj_gecerli[best_idx]} | oran={oran_gecerli[best_idx]:.4f}")

    return (board, kullanilan_max, secilen_puan, ana_dizin)

    '''
#############################################
'''
def hamle_cok_kriterli(board, tahta_puanlari2, eldeki_harfler, sozluk2,
                        w_puan=0.45, w_harf=0.20, w_dez=0.20, w_oran=0.15):
    """
    Çok-kriterli ajan:
      - ana_dizin'i oluşturur
      - min-max normalize edip +puan +harf_basi −dez −oran kompozit skor hesaplar
      - en iyi hamleyi TAHTAYA UYGULAR ve (board, kullanilan_max, secilen_puan, ana_dizin) döner

    ÖNEMLİ: 'kullanilan_max' bilgisi, seçilen hamleyi ORİJİNAL TAHTANIN KOPYASI üzerinde
    ke.kelime_yerlestir_ve_puanla5 ile bir kez DAHA çalıştırılarak alınır.
    Böylece O2'nin 'kullandı/çekti' logları kesin doğru olur.
    """
    import copy, math
    import kelimelik_engine1 as ke

    def _as_float(x):
        try:
            return float(x)
        except:
            try:
                return float(str(x).replace(",", "."))
            except:
                return 0.0

    def _minmax(vals):
        if not vals:
            return []
        lo, hi = min(vals), max(vals)
        if math.isclose(lo, hi):
            return [0.0] * len(vals)
        rng = hi - lo
        return [(v - lo) / rng for v in vals]

    # 1) Tahtadaki parçalar ve sözlük daraltma
    cikarilan = ke.extract_words(board)
    filtered_list_coklu = [item for item in cikarilan if len(item[0]) >= 2]
    filtered_list_tekli = [item for item in cikarilan if len(item[0]) == 1]
    coklu_kelimecik = [a[0] for a in filtered_list_coklu]
    tekli_kelimecik = [a[0] for a in filtered_list_tekli]

    filtreli_sozluk2 = ke.filter_scrabble_dictionary(sozluk2, coklu_kelimecik).tolist()
    filtreli_sozluk1 = ke.filter_scrabble_dictionary(sozluk2, tekli_kelimecik).tolist()

    mumkun_kelimeler  = ke.find_possible_words_and_orientations4(eldeki_harfler, filtered_list_coklu, filtreli_sozluk2)
    mumkun_kelimeler += ke.find_possible_words_and_orientations4(eldeki_harfler, filtered_list_tekli, filtreli_sozluk1)
    eldekinden_kelimeler = ke.find_valid_words_from_available_with_orientations(eldeki_harfler, sozluk2)
    mumkun_kelimeler += ke.aday_kelime_yerlestir2(board, eldekinden_kelimeler)

    # 2) Adayları simüle et
    board_temp = copy.deepcopy(board)
    yeni_tahta_puanlari = tahta_puanlari2.copy()
    yeni_tahta_puanlari[board != ''] = 0

    kelime_gecerli, xkoord_gecerli, ykoord_gecerli, orient_gecerli = [], [], [], []
    puan_gecerli, harf_basina_gecerli, dezavantaj_gecerli, oran_gecerli = [], [], [], []

    for kelime, (x_koord, y_koord), orient in mumkun_kelimeler:
        cikti = ke.kelime_yerlestir_ve_puanla5(
            kelime.upper(), x_koord, y_koord, orient,
            board_temp, yeni_tahta_puanlari, sozluk2,
            eldeki_harfler=eldeki_harfler
        )
        if not isinstance(cikti, dict):
            board_temp = copy.deepcopy(board)
            continue
        if cikti.get("gecerli"):
            puan = float(cikti["puan"])
            dez  = float(cikti["dezavantaj"])
            hb   = puan / max(len(kelime), 1)
            oran = (dez / puan) if puan > 0 else float("inf")

            kelime_gecerli.append(kelime)
            xkoord_gecerli.append(x_koord)
            ykoord_gecerli.append(y_koord)
            orient_gecerli.append(orient)
            puan_gecerli.append(puan)
            harf_basina_gecerli.append(round(hb, 2))
            dezavantaj_gecerli.append(dez)
            oran_gecerli.append(oran)

        board_temp = copy.deepcopy(board)

    # Hamle yoksa pas
    if not kelime_gecerli:
        return (board, [], 0, [])

    # 3) Kompozit skor ( +puan +harf_basi −dez −oran )
    Pn = _minmax(puan_gecerli)
    Hn = _minmax([_as_float(x) for x in harf_basina_gecerli])
    Dn = _minmax(dezavantaj_gecerli)
    Rn = _minmax(oran_gecerli)
    skor = [w_puan*Pn[i] + w_harf*Hn[i] - w_dez*Dn[i] - w_oran*Rn[i] for i in range(len(kelime_gecerli))]

    # 4) En iyiyi seç (eşitlikte puan, sonra harf_bası)
    best_idx = max(
        range(len(kelime_gecerli)),
        key=lambda i: (skor[i], puan_gecerli[i], _as_float(harf_basina_gecerli[i]))
    )

    secilen_kelime = kelime_gecerli[best_idx]
    secilen_x      = xkoord_gecerli[best_idx]
    secilen_y      = ykoord_gecerli[best_idx]
    secilen_orient = orient_gecerli[best_idx]
    secilen_puan   = int(round(puan_gecerli[best_idx]))

    # 5) (KRİTİK) KULLANILAN HARFLERİ ORİJİNAL TAHTANIN KOPYASINDA, 5'li fonksiyondan al
    temp_board = copy.deepcopy(board)
    temp_multipliers = tahta_puanlari2.copy()
    temp_multipliers[board != ''] = 0
    tmp = ke.kelime_yerlestir_ve_puanla5(
        secilen_kelime.upper(), secilen_x, secilen_y, secilen_orient,
        temp_board, temp_multipliers, sozluk2,
        eldeki_harfler=eldeki_harfler
    )
    if isinstance(tmp, dict) and "stoktan_dus" in tmp and tmp["stoktan_dus"]:
        # stoktan_dus bazen str/np.array olabilir -> listeye çevir
        km = tmp["stoktan_dus"]
        if isinstance(km, str):
            kullanilan_max = list(km)
        elif hasattr(km, "tolist"):
            kullanilan_max = list(km.tolist())
        else:
            try:
                kullanilan_max = list(km)
            except:
                kullanilan_max = []
    else:
        # nadiren boş gelirse: güvenli fallback (boş olan kareler = raftan gelenler)
        def _used_from_board(board_arr, word, x, y, orient):
            used = []
            W = word.upper()
            if str(orient).lower().startswith("h"):  # yatay
                for i, ch in enumerate(W):
                    if board_arr[x, y+i] == '' or board_arr[x, y+i] is None:
                        used.append(ch)
            else:  # düşey
                for i, ch in enumerate(W):
                    if board_arr[x+i, y] == '' or board_arr[x+i, y] is None:
                        used.append(ch)
            return used
        kullanilan_max = _used_from_board(board, secilen_kelime, secilen_x, secilen_y, secilen_orient)

    # 6) Seçimi gerçek tahtaya uygula
    _ = ke.kelime_yerlestir_ve_puanla(
        secilen_kelime.upper(), secilen_x, secilen_y, secilen_orient,
        board, yeni_tahta_puanlari, sozluk2
    )

    # 7) ana_dizin’i kompozit skora göre sırala ve döndür
    ana_dizin = list(zip(
        kelime_gecerli, xkoord_gecerli, ykoord_gecerli, orient_gecerli,
        [int(round(p)) for p in puan_gecerli],
        harf_basina_gecerli, dezavantaj_gecerli, oran_gecerli
    ))
    order = sorted(range(len(ana_dizin)),
                   key=lambda i: (skor[i], puan_gecerli[i], _as_float(harf_basina_gecerli[i])),
                   reverse=True)
    ana_dizin = [ana_dizin[i] for i in order]

    print(f"{secilen_kelime} seçildi | puan={secilen_puan} | "
          f"hb={harf_basina_gecerli[best_idx]} | dez={dezavantaj_gecerli[best_idx]} | oran={oran_gecerli[best_idx]:.4f}")

    return (board, kullanilan_max, secilen_puan, ana_dizin)

'''

####################################################################
'''
def hamle_cok_kriterli(board, tahta_puanlari2, eldeki_harfler, sozluk2,
                        w_puan=0.45, w_harf=0.20, w_dez=0.20, w_oran=0.15):
    """
    Çok-kriterli ajan:
      - ana_dizin'i oluşturur
      - min-max normalize edip +puan +harf_basi −dez −oran kompozit skor hesaplar
      - en iyi hamleyi TAHTAYA UYGULAR ve (board, kullanilan_max, secilen_puan, ana_dizin) döner

    Not: 'kullanilan_max' önce 5'li simülasyondan ('stoktan_dus') alınır;
         boş gelirse, yerleştirmeden önce tahtadaki boş karelere göre türetilir.
    """
    import copy, math
    import kelimelik_engine1 as ke

    def _as_float(x):
        try:
            return float(x)
        except:
            try:
                return float(str(x).replace(",", "."))
            except:
                return 0.0

    def _minmax(vals):
        if not vals:
            return []
        lo, hi = min(vals), max(vals)
        if math.isclose(lo, hi):
            return [0.0] * len(vals)
        rng = hi - lo
        return [(v - lo) / rng for v in vals]

    # 1) Tahtadaki parçalar ve sözlük daraltma
    cikarilan = ke.extract_words(board)
    filtered_list_coklu = [item for item in cikarilan if len(item[0]) >= 2]
    filtered_list_tekli = [item for item in cikarilan if len(item[0]) == 1]
    coklu_kelimecik = [a[0] for a in filtered_list_coklu]
    tekli_kelimecik = [a[0] for a in filtered_list_tekli]

    filtreli_sozluk2 = ke.filter_scrabble_dictionary(sozluk2, coklu_kelimecik).tolist()
    filtreli_sozluk1 = ke.filter_scrabble_dictionary(sozluk2, tekli_kelimecik).tolist()

    mumkun_kelimeler  = ke.find_possible_words_and_orientations4(eldeki_harfler, filtered_list_coklu, filtreli_sozluk2)
    mumkun_kelimeler += ke.find_possible_words_and_orientations4(eldeki_harfler, filtered_list_tekli, filtreli_sozluk1)
    eldekinden_kelimeler = ke.find_valid_words_from_available_with_orientations(eldeki_harfler, sozluk2)
    mumkun_kelimeler += ke.aday_kelime_yerlestir2(board, eldekinden_kelimeler)

    # 2) Adayları simüle et
    board_temp = copy.deepcopy(board)
    yeni_tahta_puanlari = tahta_puanlari2.copy()
    yeni_tahta_puanlari[board != ''] = 0

    kelime_gecerli, xkoord_gecerli, ykoord_gecerli, orient_gecerli = [], [], [], []
    puan_gecerli, harf_basina_gecerli, dezavantaj_gecerli, oran_gecerli = [], [], [], []

    for kelime, (x_koord, y_koord), orient in mumkun_kelimeler:
        cikti = ke.kelime_yerlestir_ve_puanla5(
            kelime.upper(), x_koord, y_koord, orient,
            board_temp, yeni_tahta_puanlari, sozluk2,
            eldeki_harfler=eldeki_harfler
        )
        if not isinstance(cikti, dict):
            board_temp = copy.deepcopy(board)
            continue
        if cikti.get("gecerli"):
            puan = float(cikti["puan"])
            dez  = float(cikti["dezavantaj"])
            hb   = puan / max(len(kelime), 1)
            oran = (dez / puan) if puan > 0 else float("inf")

            kelime_gecerli.append(kelime)
            xkoord_gecerli.append(x_koord)
            ykoord_gecerli.append(y_koord)
            orient_gecerli.append(orient)
            puan_gecerli.append(puan)
            harf_basina_gecerli.append(round(hb, 2))
            dezavantaj_gecerli.append(dez)
            oran_gecerli.append(oran)

        board_temp = copy.deepcopy(board)

    # Hamle yoksa pas
    if not kelime_gecerli:
        return (board, [], 0, [])

    # 3) Kompozit skor ( +puan +harf_basi −dez −oran )
    Pn = _minmax(puan_gecerli)
    Hn = _minmax([_as_float(x) for x in harf_basina_gecerli])
    Dn = _minmax(dezavantaj_gecerli)
    Rn = _minmax(oran_gecerli)
    skor = [w_puan*Pn[i] + w_harf*Hn[i] - w_dez*Dn[i] - w_oran*Rn[i] for i in range(len(kelime_gecerli))]

    # 4) En iyiyi seç (eşitlikte puan, sonra harf_bası)
    best_idx = max(
        range(len(kelime_gecerli)),
        key=lambda i: (skor[i], puan_gecerli[i], _as_float(harf_basina_gecerli[i]))
    )

    secilen_kelime = kelime_gecerli[best_idx]
    secilen_x      = xkoord_gecerli[best_idx]
    secilen_y      = ykoord_gecerli[best_idx]
    secilen_orient = orient_gecerli[best_idx]
    secilen_puan   = int(round(puan_gecerli[best_idx]))

    # 5) KULLANILAN HARFLERİ güvenle elde et:
    # 5a) Önce orijinal tahtanın KOPYASINDA 5'li fonksiyonla stoktan_dus'u dene
    temp_board = copy.deepcopy(board)
    temp_mult  = tahta_puanlari2.copy()
    temp_mult[board != ''] = 0
    tmp = ke.kelime_yerlestir_ve_puanla5(
        secilen_kelime.upper(), secilen_x, secilen_y, secilen_orient,
        temp_board, temp_mult, sozluk2,
        eldeki_harfler=eldeki_harfler
    )
    kullanilan_max = []
    if isinstance(tmp, dict) and "stoktan_dus" in tmp:
        km = tmp["stoktan_dus"]
        if km:
            # km tipini güvenle liste yapalım
            if isinstance(km, str):
                kullanilan_max = list(km)
            elif hasattr(km, "tolist"):
                kullanilan_max = list(km.tolist())
            else:
                try:
                    kullanilan_max = list(km)
                except:
                    kullanilan_max = []

    # 5b) Hâlâ boşsa, tahtadaki BOŞ kareleri kontrol ederek türet
    if not kullanilan_max:
        def _is_empty(cell):
            # Boş hücre: '', ' ' veya None (falsy)
            return (cell is None) or (cell == '') or (cell == ' ') or (not bool(cell))
        def _used_from_board(board_arr, word, x, y, orient):
            used = []
            W = word.upper()
            if str(orient).lower().startswith("h"):  # yatay
                for i, ch in enumerate(W):
                    if _is_empty(board_arr[x, y+i]):
                        used.append(ch)
            else:  # düşey
                for i, ch in enumerate(W):
                    if _is_empty(board_arr[x+i, y]):
                        used.append(ch)
            return used
        kullanilan_max = _used_from_board(board, secilen_kelime, secilen_x, secilen_y, secilen_orient)

    # 6) Seçimi gerçek tahtaya uygula
    _ = ke.kelime_yerlestir_ve_puanla(
        secilen_kelime.upper(), secilen_x, secilen_y, secilen_orient,
        board, yeni_tahta_puanlari, sozluk2
    )

    # 7) ana_dizin’i kompozit skora göre sırala ve döndür
    ana_dizin = list(zip(
        kelime_gecerli, xkoord_gecerli, ykoord_gecerli, orient_gecerli,
        [int(round(p)) for p in puan_gecerli],
        harf_basina_gecerli, dezavantaj_gecerli, oran_gecerli
    ))
    order = sorted(range(len(ana_dizin)),
                   key=lambda i: (skor[i], puan_gecerli[i], _as_float(harf_basina_gecerli[i])),
                   reverse=True)
    ana_dizin = [ana_dizin[i] for i in order]

    print(f"{secilen_kelime} seçildi | puan={secilen_puan} | "
          f"hb={harf_basina_gecerli[best_idx]} | dez={dezavantaj_gecerli[best_idx]} | oran={oran_gecerli[best_idx]:.4f}")

    return (board, kullanilan_max, secilen_puan, ana_dizin)
'''
#############################################################################

# kelimelik_engine1.py dosyasına eklenecek veya mevcutlarıyla değiştirilecek optimize fonksiyonlar:

from collections import Counter
from itertools import permutations

#def filter_dictionary(dictionary, available_letters, max_length):
 #   harf_seti = set(available_letters)
 #   return [w for w in dictionary if len(w) <= max_length and all(h in harf_seti for h in w)]

def filter_dictionary(dictionary, available_letters, max_length=None):
    harf_seti = set(available_letters)
    if max_length:
        return [w for w in dictionary if len(w) <= max_length and all(h in harf_seti for h in w)]
    else:
        return [w for w in dictionary if all(h in harf_seti for h in w)]

def hamle_cok_kriterli_eski2(board, tahta_puanlari2, eldeki_harfler, sozluk2,
                        w_puan=0.45, w_harf=0.20, w_dez=0.20, w_oran=0.15):
    import copy, math
    import kelimelik_engine1 as ke

    def _as_float(x):
        try: return float(x)
        except: return 0.0

    def _minmax(vals):
        if not vals: return []
        lo, hi = min(vals), max(vals)
        if math.isclose(lo, hi): return [0.0] * len(vals)
        rng = hi - lo
        return [(v - lo) / rng for v in vals]

    cikarilan = ke.extract_words(board)
    filtered_list_coklu = [item for item in cikarilan if len(item[0]) >= 2]
    filtered_list_tekli = [item for item in cikarilan if len(item[0]) == 1]
    coklu_kelimecik = [a[0] for a in filtered_list_coklu]
    tekli_kelimecik = [a[0] for a in filtered_list_tekli]

    # >>> EKLE (tip güvenliği)
    coklu_kelimecik = [str(x) for x in coklu_kelimecik if str(x)]
    tekli_kelimecik = [str(x) for x in tekli_kelimecik if str(x)]

    filtreli_sozluk2 = ke.filter_scrabble_dictionary(sozluk2, coklu_kelimecik).tolist()
    filtreli_sozluk1 = ke.filter_scrabble_dictionary(sozluk2, tekli_kelimecik).tolist()
    #kelime_uzunluk = min(0, len(eldeki_harfler)+5)
    filtreli_sozluk2 = filter_dictionary(filtreli_sozluk2, eldeki_harfler)
    filtreli_sozluk1 = filter_dictionary(filtreli_sozluk1, eldeki_harfler)

    mumkun_kelimeler  = ke.find_possible_words_and_orientations4(eldeki_harfler, filtered_list_coklu, filtreli_sozluk2)
    mumkun_kelimeler += ke.find_possible_words_and_orientations4(eldeki_harfler, filtered_list_tekli, filtreli_sozluk1)
    eldekinden_kelimeler = ke.find_valid_words_from_available_with_orientations(eldeki_harfler, sozluk2)
    mumkun_kelimeler += ke.aday_kelime_yerlestir2(board, eldekinden_kelimeler)

    # Hamle sayısı çoksa kısıtla
    #if len(mumkun_kelimeler) > 300:
    #    mumkun_kelimeler = sorted(mumkun_kelimeler, key=lambda x: -len(x[0]))[:300]

    board_temp = copy.deepcopy(board)
    yeni_tahta_puanlari = tahta_puanlari2.copy()
    yeni_tahta_puanlari[board != ''] = 0

    kelime_gecerli, xkoord_gecerli, ykoord_gecerli, orient_gecerli = [], [], [], []
    puan_gecerli, harf_basina_gecerli, dezavantaj_gecerli, oran_gecerli = [], [], [], []

    for kelime, (x_koord, y_koord), orient in mumkun_kelimeler:
        cikti = ke.kelime_yerlestir_ve_puanla5(
            kelime.upper(), x_koord, y_koord, orient,
            board_temp, yeni_tahta_puanlari, sozluk2,
            eldeki_harfler=eldeki_harfler
        )
        if not isinstance(cikti, dict):
            board_temp = copy.deepcopy(board)
            continue
        if cikti.get("gecerli"):
            puan = float(cikti["puan"])
            dez  = float(cikti["dezavantaj"])
            hb   = puan / max(len(kelime), 1)
            oran = (dez / puan) if puan > 0 else float("inf")
            oran=round(oran,2)

            kelime_gecerli.append(kelime)
            xkoord_gecerli.append(x_koord)
            ykoord_gecerli.append(y_koord)
            orient_gecerli.append(orient)
            puan_gecerli.append(puan)
            harf_basina_gecerli.append(round(hb, 2))
            dezavantaj_gecerli.append(dez)
            oran_gecerli.append(oran)

        board_temp = copy.deepcopy(board)

    if not kelime_gecerli:
        return (board, [], 0, [])

    Pn = _minmax(puan_gecerli)
    Hn = _minmax([_as_float(x) for x in harf_basina_gecerli])
    Dn = _minmax(dezavantaj_gecerli)
    Rn = _minmax(oran_gecerli)
    skor = [w_puan*Pn[i] + w_harf*Hn[i] - w_dez*Dn[i] - w_oran*Rn[i] for i in range(len(kelime_gecerli))]

    best_idx = max(
        range(len(kelime_gecerli)),
        key=lambda i: (skor[i], puan_gecerli[i], _as_float(harf_basina_gecerli[i]))
    )

    secilen_kelime = kelime_gecerli[best_idx]
    secilen_x      = xkoord_gecerli[best_idx]
    secilen_y      = ykoord_gecerli[best_idx]
    secilen_orient = orient_gecerli[best_idx]
    secilen_puan   = int(round(puan_gecerli[best_idx]))

    tmp = ke.kelime_yerlestir_ve_puanla5(
        secilen_kelime.upper(), secilen_x, secilen_y, secilen_orient,
        copy.deepcopy(board), yeni_tahta_puanlari, sozluk2,
        eldeki_harfler=eldeki_harfler
    )
    kullanilan_max = tmp.get("stoktan_dus", []) if isinstance(tmp, dict) else []

    _ = ke.kelime_yerlestir_ve_puanla(
        secilen_kelime.upper(), secilen_x, secilen_y, secilen_orient,
        board, yeni_tahta_puanlari, sozluk2
    )

    ana_dizin = list(zip(
        kelime_gecerli, xkoord_gecerli, ykoord_gecerli, orient_gecerli,
        [int(round(p)) for p in puan_gecerli],
        harf_basina_gecerli, dezavantaj_gecerli, oran_gecerli
    ))
    order = sorted(range(len(ana_dizin)),
                   key=lambda i: (skor[i], puan_gecerli[i], _as_float(harf_basina_gecerli[i])),
                   reverse=True)
    ana_dizin = [ana_dizin[i] for i in order]

    print(f"{secilen_kelime} seçildi | puan={secilen_puan} | "
          f"hb={harf_basina_gecerli[best_idx]} | dez={dezavantaj_gecerli[best_idx]} | oran={oran_gecerli[best_idx]:.4f}")

    return (board, kullanilan_max, secilen_puan, ana_dizin)
################################################################
'''
# orijinal tanımın yerine / içine koy
def filter_scrabble_dictionary(sozluk, words):
    import numpy as np

    # 0) Boş veya None gelirse sözlüğü aynen döndür (erken çıkış)
    if words is None:
        return np.array(sozluk, dtype=str)
    try:
        # 1) Tüm öğeleri DÜZ string’e çevir (tuple/list vs. gelebiliyor)
        words_str = [str(w) for w in words if str(w)]
        if len(words_str) == 0:
            return np.array(sozluk, dtype=str)

        # 2) Güvenli şekilde upper (Türkçe karakterleri zaten büyük harfli tutuyorsan sorun yok)
        words_upper = np.char.upper(np.array(words_str, dtype='U'))  # 'U' = unicode string dtype
        sozluk_upper = np.char.upper(np.array(sozluk, dtype='U'))

        # ... senin mevcut filtreleme mantığın devam ...
        # ÖRN: sadece örnek (senin orijinal eşleştirme / filtreleme kriterin neyse onu uygula):
        mask = np.isin(sozluk_upper, words_upper)
        return np.array(sozluk)[mask]

    except Exception as e:
        # Beklenmedik bir tip karışıklığında güvenli geri dönüş
        # (istersen loglayabilirsin)
        return np.array(sozluk, dtype=str)
'''
####################################################
def filter_scrabble_dictionary(sozluk, words):
    """
    Kelime listesini filtreleyen güvenli versiyon.
    - words içindeki parçaları büyük harfe çevirerek
    - sozluk içindeki kelimeleri eşleştirip
    - eşleşenleri geri döndürür
    """
    import numpy as np

    def _to_uarray(seq):
        if seq is None:
            seq = []
        # Her elemanı str'e çevir, unicode string numpy array'e dönüştür
        return np.array([str(x) for x in seq], dtype='U')

    # Güvenli string dizilerine dönüştür
    words_arr  = _to_uarray(words)
    sozluk_arr = _to_uarray(sozluk)

    # Büyük harfe çevir
    words_upper  = np.char.upper(words_arr)
    sozluk_upper = np.char.upper(sozluk_arr)

    # En az 1 harfli kelime parçaları al (boş stringler filtrelenir)
    valid_parts = [part for part in np.unique(words_upper) if len(part) >= 1]
    clean_sozluk = [kelime for kelime in np.unique(sozluk_upper) if len(kelime) >= 1]

    # Eşleşenleri bul
    filtered = []
    for kelime in clean_sozluk:
        if any(part in kelime for part in valid_parts):
            filtered.append(kelime)

    return np.sort(np.unique(filtered))

#############################################################################

# Dosyanın başına veya env.step() içine:
import collections

def raftan_cikar(raf, eksilen):
    raf_counter = collections.Counter(raf)
    eksilen_counter = collections.Counter(eksilen)
    kalan = raf_counter - eksilen_counter
    return list(kalan.elements())

##############################################################################

def hamle_cok_kriterli_eski(board, tahta_puanlari2, eldeki_harfler, sozluk2,
                        w_puan=0.45, w_harf=0.20, w_dez=0.20, w_oran=0.15):
    import copy, math
    import kelimelik_engine1 as ke

    def _as_float(x):
        try: return float(x)
        except: return 0.0

    def _minmax(vals):
        if not vals: return []
        lo, hi = min(vals), max(vals)
        if math.isclose(lo, hi): return [0.0] * len(vals)
        rng = hi - lo
        return [(v - lo) / rng for v in vals]

    # 1. TAHTADAN kelimecikleri çıkar
    cikarilan = ke.extract_words(board)
    filtered_list_coklu = [item for item in cikarilan if len(item[0]) >= 2]
    filtered_list_tekli = [item for item in cikarilan if len(item[0]) == 1]

    coklu_kelimecik = [str(x[0]) for x in filtered_list_coklu if str(x[0])]
    tekli_kelimecik = [str(x[0]) for x in filtered_list_tekli if str(x[0])]

    # 2. Sözlük filtrele
    filtreli_sozluk2 = ke.filter_scrabble_dictionary(sozluk2, coklu_kelimecik).tolist()
    filtreli_sozluk1 = ke.filter_scrabble_dictionary(sozluk2, tekli_kelimecik).tolist()

    kelime_uzunluk = len(eldeki_harfler) + 2
    filtreli_sozluk2 = filter_dictionary(filtreli_sozluk2, eldeki_harfler)
    filtreli_sozluk1 = filter_dictionary(filtreli_sozluk1, eldeki_harfler)

    # 3. Mümkün kelimeleri üret
    mumkun_kelimeler  = ke.find_possible_words_and_orientations4(eldeki_harfler, filtered_list_coklu, filtreli_sozluk2)
    mumkun_kelimeler += ke.find_possible_words_and_orientations4(eldeki_harfler, filtered_list_tekli, filtreli_sozluk1)
    eldekinden_kelimeler = ke.find_valid_words_from_available_with_orientations(eldeki_harfler, sozluk2)
    mumkun_kelimeler += ke.aday_kelime_yerlestir2(board, eldekinden_kelimeler)

    # >>> DEBUG
    print(f"[DEBUG] Toplam aday kelime sayısı: {len(mumkun_kelimeler)}")
    if len(mumkun_kelimeler) == 0:
        print(f"[UYARI] Hiç kelime üretilemedi | Raf: {eldeki_harfler}")

    # 4. Puanla ve filtrele
    board_temp = copy.deepcopy(board)
    yeni_tahta_puanlari = tahta_puanlari2.copy()
    yeni_tahta_puanlari[board != ''] = 0

    kelime_gecerli, xkoord_gecerli, ykoord_gecerli, orient_gecerli = [], [], [], []
    puan_gecerli, harf_basina_gecerli, dezavantaj_gecerli, oran_gecerli = [], [], [], []

    for kelime, (x_koord, y_koord), orient in mumkun_kelimeler:
        cikti = ke.kelime_yerlestir_ve_puanla5(
            kelime.upper(), x_koord, y_koord, orient,
            board_temp, yeni_tahta_puanlari, sozluk2,
            eldeki_harfler=eldeki_harfler
        )
        if not isinstance(cikti, dict):
            continue
        if cikti.get("gecerli"):
            puan = float(cikti["puan"])
            dez  = float(cikti["dezavantaj"])
            hb   = puan / max(len(kelime), 1)
            oran = round((dez / puan), 2) if puan > 0 else float("inf")

            kelime_gecerli.append(kelime)
            xkoord_gecerli.append(x_koord)
            ykoord_gecerli.append(y_koord)
            orient_gecerli.append(orient)
            puan_gecerli.append(puan)
            harf_basina_gecerli.append(round(hb, 2))
            dezavantaj_gecerli.append(dez)
            oran_gecerli.append(oran)

    if not kelime_gecerli:
        print(f"[UYARI] Hiç GEÇERLİ kelime bulunamadı | Raf: {eldeki_harfler} | Aday sayısı: {len(mumkun_kelimeler)}")
        return (board, [], 0, [])

    Pn = _minmax(puan_gecerli)
    Hn = _minmax(harf_basina_gecerli)
    Dn = _minmax(dezavantaj_gecerli)
    Rn = _minmax(oran_gecerli)
    skor = [w_puan*Pn[i] + w_harf*Hn[i] - w_dez*Dn[i] - w_oran*Rn[i] for i in range(len(kelime_gecerli))]

    best_idx = max(
        range(len(kelime_gecerli)),
        key=lambda i: (skor[i], puan_gecerli[i], _as_float(harf_basina_gecerli[i]))
    )

    secilen_kelime = kelime_gecerli[best_idx]
    secilen_x      = xkoord_gecerli[best_idx]
    secilen_y      = ykoord_gecerli[best_idx]
    secilen_orient = orient_gecerli[best_idx]
    secilen_puan   = int(round(puan_gecerli[best_idx]))

    tmp = ke.kelime_yerlestir_ve_puanla5(
        secilen_kelime.upper(), secilen_x, secilen_y, secilen_orient,
        copy.deepcopy(board), yeni_tahta_puanlari, sozluk2,
        eldeki_harfler=eldeki_harfler
    )
    kullanilan_max = tmp.get("stoktan_dus", []) if isinstance(tmp, dict) else []

    _ = ke.kelime_yerlestir_ve_puanla(
        secilen_kelime.upper(), secilen_x, secilen_y, secilen_orient,
        board, yeni_tahta_puanlari, sozluk2
    )

    ana_dizin = list(zip(
        kelime_gecerli, xkoord_gecerli, ykoord_gecerli, orient_gecerli,
        [int(round(p)) for p in puan_gecerli],
        harf_basina_gecerli, dezavantaj_gecerli, oran_gecerli
    ))
    order = sorted(range(len(ana_dizin)),
                   key=lambda i: (skor[i], puan_gecerli[i], _as_float(harf_basina_gecerli[i])),
                   reverse=True)
    ana_dizin = [ana_dizin[i] for i in order]

    print(f"{secilen_kelime} seçildi | puan={secilen_puan} | hb={harf_basina_gecerli[best_idx]} | dez={dezavantaj_gecerli[best_idx]} | oran={oran_gecerli[best_idx]:.2f}")

    return (board, kullanilan_max, secilen_puan, ana_dizin)

######################################################################################

def hamle_cok_kriterli_simdilik_eski_ama_emektar(board, tahta_puanlari2, eldeki_harfler, sozluk2,
                             w_puan=0.45, w_harf=0.20, w_dez=0.20, w_oran=0.15):
    import copy, math
    import kelimelik_engine1 as ke
    #DAWG EKLEMESİ
    import dawg_helper as dh

    #DAWG EKLEMESİ
    # 1) DAWG'ı bir kere kur (dışarıda, cache'li)
    dawg = dh.build_dawg_from_dictionary(sozluk2)
    #DAWG EKLEMESİ
    # 2) Tahtadan parçaları çıkar
    word_parts = dh.extract_word_parts(board)

    #DAWG EKLEMESİ
    # 3) Her parça için DAWG adaylarını üret
    candidates = []
    for wp in word_parts:
        candidates.extend(dh.generate_candidates_for_word_part_with_board(dawg, board, wp, eldeki_harfler))


    # --- Yardımcılar: sadece yerel, engine'e dokunmuyoruz ---
    def _as_float(x):
        try: 
            return float(x)
        except: 
            return 0.0

    def _minmax(vals):
        # Boş listeyse boş dön
        if not vals:
            return []
        # NaN/inf temizliği
        clean = []
        for v in vals:
            if v is None:
                clean.append(0.0)
                continue
            try:
                fv = float(v)
                if math.isinf(fv) or math.isnan(fv):
                    clean.append(0.0)
                else:
                    clean.append(fv)
            except:
                clean.append(0.0)
        lo, hi = min(clean), max(clean)
        # Tekillik: bütün değerler aynı → nötr 0.5
        if math.isclose(lo, hi):
            return [0.5] * len(clean)
        rng = hi - lo
        return [(v - lo) / rng for v in clean]

    # --- 1) Tahtadan kelimecikleri çıkar + sözlük daralt ---
    cikarilan = ke.extract_words(board)
    filtered_list_coklu = [item for item in cikarilan if len(item[0]) >= 2]
    filtered_list_tekli = [item for item in cikarilan if len(item[0]) == 1]
    coklu_kelimecik = [str(a[0]) for a in filtered_list_coklu if str(a[0])]
    tekli_kelimecik = [str(a[0]) for a in filtered_list_tekli if str(a[0])]

    filtreli_sozluk2 = ke.filter_scrabble_dictionary(sozluk2, coklu_kelimecik).tolist()
    filtreli_sozluk1 = ke.filter_scrabble_dictionary(sozluk2, tekli_kelimecik).tolist()

    # Not: Burada sadece mevcut helper'ı kullanıyoruz (filter_dictionary zaten vardı).
    # Eğer filter_dictionary tahtadaki harfleri hesaba katmıyorsa, aynı davranışı koruyoruz.
    #filtreli_sozluk2 = filter_dictionary(filtreli_sozluk2, eldeki_harfler)
    #filtreli_sozluk1 = filter_dictionary(filtreli_sozluk1, eldeki_harfler)

    # --- 2) Aday üretimi (engine fonksiyonları) ---
    mumkun_kelimeler  = ke.find_possible_words_and_orientations4(eldeki_harfler, filtered_list_coklu, filtreli_sozluk2)
    mumkun_kelimeler += ke.find_possible_words_and_orientations4(eldeki_harfler, filtered_list_tekli, filtreli_sozluk1)
    eldekinden_kelimeler = ke.find_valid_words_from_available_with_orientations(eldeki_harfler, sozluk2)
    mumkun_kelimeler += ke.aday_kelime_yerlestir2(board, eldekinden_kelimeler)

    # --- 3) Puanlama için board/bouns kopyası ---
    board_temp = copy.deepcopy(board)
    yeni_tahta_puanlari = tahta_puanlari2.copy()
    yeni_tahta_puanlari[board != ''] = 0

    kelime_gecerli, xkoord_gecerli, ykoord_gecerli, orient_gecerli = [], [], [], []
    puan_gecerli, harf_basina_gecerli, dezavantaj_gecerli, oran_gecerli = [], [], [], []

    # --- 4) Her aday için engine'in yerleştir & puanla fonksiyonuyla kontrol ---
    for kelime, (x_koord, y_koord), orient in mumkun_kelimeler:
        #orient_hv = orient[0].lower() if isinstance(orient, str) else orient
        if isinstance(orient, str):
            orient_hv = orient[0].lower()   # 'h' ya da 'v'
        else:
            orient_hv = orient              # zaten 'h'/'v' ise dokunma


        
        cikti = ke.kelime_yerlestir_ve_puanla5(
            kelime.upper(), x_koord, y_koord, orient_hv,
            board_temp, yeni_tahta_puanlari, sozluk2,
            eldeki_harfler=eldeki_harfler
        )
        if not isinstance(cikti, dict):
            board_temp = copy.deepcopy(board)
            continue

        if cikti.get("gecerli"):
            puan = float(cikti.get("puan", 0.0))
            dez  = float(cikti.get("dezavantaj", 0.0))

            # HB ve ORAN güvenli hesap
            hb = puan / max(len(kelime), 1)
            # 0-bölme koruması + üst limit → inf/NaN engeli
            oran_raw = (dez / puan) if puan > 0 else 0.0
            if math.isnan(oran_raw) or math.isinf(oran_raw):
                oran_raw = 0.0
            # Çok uç durumları normalize etmek için üst limit (ör. 10)
            oran = round(min(oran_raw, 10.0), 2)

            kelime_gecerli.append(kelime)
            xkoord_gecerli.append(x_koord)
            ykoord_gecerli.append(y_koord)
            orient_gecerli.append(orient_hv)
            puan_gecerli.append(puan)
            harf_basina_gecerli.append(round(hb, 2))
            dezavantaj_gecerli.append(dez)
            oran_gecerli.append(oran)

        board_temp = copy.deepcopy(board)

    # Hiç geçerli aday yoksa—orijinal davranışı koruyoruz (engine dışı fallback yok)
    if not kelime_gecerli:
        return (board, [], 0, [])

    # --- 5) Normalize + skor ---
    Pn = _minmax(puan_gecerli)
    Hn = _minmax([_as_float(x) for x in harf_basina_gecerli])
    Dn = _minmax(dezavantaj_gecerli)
    Rn = _minmax(oran_gecerli)

    skor = []
    for i in range(len(kelime_gecerli)):
        s = (w_puan * Pn[i]) + (w_harf * Hn[i]) - (w_dez * Dn[i]) - (w_oran * Rn[i])
        # NaN/inf koruması
        if math.isnan(s) or math.isinf(s):
            s = 0.0
        skor.append(s)

    # Skor vektörü bozulduysa (hepsi aynı veya tümü 0.0) → puana fallback
    # (deterministik ajanın davranışına yakınlaştırır)
    all_same = all(math.isclose(skor[0], s) for s in skor)
    all_zero = all(math.isclose(s, 0.0) for s in skor)
    if all_same or all_zero:
        best_idx = max(range(len(puan_gecerli)), key=lambda i: (puan_gecerli[i], _as_float(harf_basina_gecerli[i])))
    else:
        best_idx = max(
            range(len(kelime_gecerli)),
            key=lambda i: (skor[i], puan_gecerli[i], _as_float(harf_basina_gecerli[i]))
        )

    secilen_kelime = kelime_gecerli[best_idx]
    secilen_x      = xkoord_gecerli[best_idx]
    secilen_y      = ykoord_gecerli[best_idx]
    secilen_orient = orient_gecerli[best_idx]
    secilen_puan   = int(round(puan_gecerli[best_idx]))

    # Engine içindeki aynı akış (stoktan düşecek harfleri almak için)
    tmp = ke.kelime_yerlestir_ve_puanla5(
        secilen_kelime.upper(), secilen_x, secilen_y, secilen_orient,
        copy.deepcopy(board), yeni_tahta_puanlari, sozluk2,
        eldeki_harfler=eldeki_harfler
    )
    kullanilan_max = tmp.get("stoktan_dus", []) if isinstance(tmp, dict) else []

    _ = ke.kelime_yerlestir_ve_puanla(
        secilen_kelime.upper(), secilen_x, secilen_y, secilen_orient,
        board, yeni_tahta_puanlari, sozluk2
    )

    ana_dizin = list(zip(
        kelime_gecerli, xkoord_gecerli, ykoord_gecerli, orient_gecerli,
        [int(round(p)) for p in puan_gecerli],
        harf_basina_gecerli, dezavantaj_gecerli, oran_gecerli
    ))

    order = sorted(
        range(len(ana_dizin)),
        key=lambda i: (skor[i], puan_gecerli[i], _as_float(harf_basina_gecerli[i])),
        reverse=True
    )
    ana_dizin = [ana_dizin[i] for i in order]

    print(f"{secilen_kelime} seçildi | puan={secilen_puan} | "
          f"hb={harf_basina_gecerli[best_idx]} | dez={dezavantaj_gecerli[best_idx]} | oran={oran_gecerli[best_idx]:.4f}")

    return (board, kullanilan_max, secilen_puan, ana_dizin)

################################################################################################

def hamle_cok_kriterli(board, tahta_puanlari2, eldeki_harfler, sozluk2,
                       w_puan=0.45, w_harf=0.20, w_dez=0.20, w_oran=0.15):
    import copy, math
    import kelimelik_engine1 as ke
    import dawg_helper as dh

    # --- Yardımcılar: sadece yerel, engine'e dokunmuyoruz ---
    def _as_float(x):
        try:
            return float(x)
        except:
            return 0.0

    def _minmax(vals):
        if not vals:
            return []
        clean = []
        for v in vals:
            if v is None:
                clean.append(0.0)
                continue
            try:
                fv = float(v)
                if math.isinf(fv) or math.isnan(fv):
                    clean.append(0.0)
                else:
                    clean.append(fv)
            except:
                clean.append(0.0)

        lo, hi = min(clean), max(clean)
        if math.isclose(lo, hi):
            return [0.5] * len(clean)
        rng = hi - lo
        return [(v - lo) / rng for v in clean]

    # --- 0) DAWG cache (sözlük değişmedikçe yeniden kurma) ---
    # sozluk2 genelde sabit; bu yüzden cache ciddi hız kazandırır.
    cache_key = (id(sozluk2), len(sozluk2))
    if not hasattr(hamle_cok_kriterli, "_dawg_cache"):
        hamle_cok_kriterli._dawg_cache = {}

    if cache_key not in hamle_cok_kriterli._dawg_cache:
        # sözlüğün senin tarafta zaten UPPERCASE olduğunu varsayıyorum;
        # değilse burada .upper() ile normalize edebilirsin.
        hamle_cok_kriterli._dawg_cache[cache_key] = dh.build_dawg_from_dictionary(sozluk2)

    dawg = hamle_cok_kriterli._dawg_cache[cache_key]

    # --- 1) Tahtadan kelimecikleri çıkar (senin engine fonksiyonun) ---
    # ke.extract_words: [(kelime, (x,y), 'horizontal'/'vertical'), ...]
    cikarilan = ke.extract_words(board)

    # board boş mu?
    try:
        board_has_letters = bool((board != '').any())
    except Exception:
        board_has_letters = any(cell != '' for row in board for cell in row)

    # --- 2) Aday üretimi ---
    # ÇIKTI FORMATINI KORUYORUZ:
    # mumkun_kelimeler elemanı: (kelime, (x,y), orient)  orient: 'h'/'v' veya 'horizontal'/'vertical' değil,
    # senin aşağıdaki loop'unun beklediği gibi ilk harfiyle 'h'/'v' çıkarılabilecek şekilde 'h'/'v' veriyoruz.
    mumkun_set = set()

    if board_has_letters:
        # 2a) DAWG: tahtadaki her kelimecikten prefix+suffix ile genişlet
        # dh.generate_candidates_for_word_part_with_board bekler: (part,(y,x),'H'/'V')
        for part, (x0, y0), orient in cikarilan:
            if not part:
                continue

            part_up = str(part).upper()

            yon = 'H' if str(orient).lower().startswith('h') else 'V'
            wp = (part_up, (int(y0), int(x0)), yon)

            cand_list = dh.generate_candidates_for_word_part_with_board(
                dawg=dawg,
                board=board,
                word_part=wp,
                rack_letters=eldeki_harfler
            )
            # cand_list: [(kelime, x_start, y_start, 'H'/'V'), ...]
            for w, xs, ys, yon2 in cand_list:
                orient_hv = 'h' if str(yon2).upper() == 'H' else 'v'
                mumkun_set.add((str(w).upper(), (int(xs), int(ys)), orient_hv))

        # İstersen burada tek harf parçalarını da ayrıca eklemek mümkün;
        # ama extract_words zaten tek harf kelimecik döndürüyor.
        mumkun_kelimeler = list(mumkun_set)

    else:
        # 2b) Tahta boşsa: eski davranışı koru (ilk hamle)
        eldekinden_kelimeler = ke.find_valid_words_from_available_with_orientations(eldeki_harfler, sozluk2)
        mumkun_kelimeler = ke.aday_kelime_yerlestir2(board, eldekinden_kelimeler)

    # --- 3) Puanlama için board/bonus kopyası ---
    board_temp = copy.deepcopy(board)
    yeni_tahta_puanlari = tahta_puanlari2.copy()
    yeni_tahta_puanlari[board != ''] = 0

    kelime_gecerli, xkoord_gecerli, ykoord_gecerli, orient_gecerli = [], [], [], []
    puan_gecerli, harf_basina_gecerli, dezavantaj_gecerli, oran_gecerli = [], [], [], []

    # --- 4) Her aday için engine'in yerleştir & puanla fonksiyonuyla kontrol ---
    for kelime, (x_koord, y_koord), orient in mumkun_kelimeler:
        if isinstance(orient, str):
            orient_hv = orient[0].lower()   # 'h' ya da 'v'
        else:
            orient_hv = orient              # zaten 'h'/'v' ise dokunma

        cikti = ke.kelime_yerlestir_ve_puanla5(
            kelime.upper(), x_koord, y_koord, orient_hv,
            board_temp, yeni_tahta_puanlari, sozluk2,
            eldeki_harfler=eldeki_harfler
        )

        if not isinstance(cikti, dict):
            board_temp = copy.deepcopy(board)
            continue

        if cikti.get("gecerli"):
            # DİKKAT: senin engine çıktın bu anahtarları kullanıyor: "puan", "dezavantaj", "stoktan_dus"
            puan = float(cikti.get("puan", 0.0))
            dez  = float(cikti.get("dezavantaj", 0.0))

            hb = puan / max(len(kelime), 1)

            oran_raw = (dez / puan) if puan > 0 else 0.0
            if math.isnan(oran_raw) or math.isinf(oran_raw):
                oran_raw = 0.0
            oran = round(min(oran_raw, 10.0), 2)

            kelime_gecerli.append(kelime)
            xkoord_gecerli.append(x_koord)
            ykoord_gecerli.append(y_koord)
            orient_gecerli.append(orient_hv)
            puan_gecerli.append(puan)
            harf_basina_gecerli.append(round(hb, 2))
            dezavantaj_gecerli.append(dez)
            oran_gecerli.append(oran)

        board_temp = copy.deepcopy(board)

    if not kelime_gecerli:
        return (board, [], 0, [])

    # --- 5) Normalize + skor ---
    Pn = _minmax(puan_gecerli)
    Hn = _minmax([_as_float(x) for x in harf_basina_gecerli])
    Dn = _minmax(dezavantaj_gecerli)
    Rn = _minmax(oran_gecerli)

    skor = []
    for i in range(len(kelime_gecerli)):
        s = (w_puan * Pn[i]) + (w_harf * Hn[i]) - (w_dez * Dn[i]) - (w_oran * Rn[i])
        if math.isnan(s) or math.isinf(s):
            s = 0.0
        skor.append(s)

    all_same = all(math.isclose(skor[0], s) for s in skor)
    all_zero = all(math.isclose(s, 0.0) for s in skor)

    if all_same or all_zero:
        best_idx = max(
            range(len(puan_gecerli)),
            key=lambda i: (puan_gecerli[i], _as_float(harf_basina_gecerli[i]))
        )
    else:
        best_idx = max(
            range(len(kelime_gecerli)),
            key=lambda i: (skor[i], puan_gecerli[i], _as_float(harf_basina_gecerli[i]))
        )

    secilen_kelime = kelime_gecerli[best_idx]
    secilen_x      = xkoord_gecerli[best_idx]
    secilen_y      = ykoord_gecerli[best_idx]
    secilen_orient = orient_gecerli[best_idx]
    secilen_puan   = int(round(puan_gecerli[best_idx]))

    # --- 6) Stoktan düşecek harfleri almak için (senin mevcut akışını koruyoruz) ---
    tmp = ke.kelime_yerlestir_ve_puanla5(
        secilen_kelime.upper(), secilen_x, secilen_y, secilen_orient,
        copy.deepcopy(board), yeni_tahta_puanlari, sozluk2,
        eldeki_harfler=eldeki_harfler
    )
    kullanilan_max = tmp.get("stoktan_dus", []) if isinstance(tmp, dict) else []

    _ = ke.kelime_yerlestir_ve_puanla(
        secilen_kelime.upper(), secilen_x, secilen_y, secilen_orient,
        board, yeni_tahta_puanlari, sozluk2
    )

    # --- 7) ana_dizin (senin format) ---
    ana_dizin = list(zip(
        kelime_gecerli, xkoord_gecerli, ykoord_gecerli, orient_gecerli,
        [int(round(p)) for p in puan_gecerli],
        harf_basina_gecerli, dezavantaj_gecerli, oran_gecerli
    ))

    order = sorted(
        range(len(ana_dizin)),
        key=lambda i: (skor[i], puan_gecerli[i], _as_float(harf_basina_gecerli[i])),
        reverse=True
    )
    ana_dizin = [ana_dizin[i] for i in order]

    #print(f"{secilen_kelime} seçildi | puan={secilen_puan} | "
          #f"hb={harf_basina_gecerli[best_idx]} | dez={dezavantaj_gecerli[best_idx]} | oran={oran_gecerli[best_idx]:.4f}")

    return (board, kullanilan_max, secilen_puan, ana_dizin)
