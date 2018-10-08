subroutine binned_map(signal, pixidx, mappixels, hits)
    implicit none

    real(kind=8), dimension(:), intent(in) :: signal
    integer(kind=8), dimension(size(signal)), intent(in) :: pixidx
    real(kind=8), dimension(:), intent(inout) :: mappixels
    integer(kind=8), dimension(size(mappixels)), intent(inout) :: hits

    integer :: i

    mappixels = 0.0
    hits = 0

    do i = 1, size(signal)
        mappixels(pixidx(i) + 1) = mappixels(pixidx(i) + 1) + signal(i)
        hits(pixidx(i) + 1) = hits(pixidx(i) + 1) + 1
    end do

    do i = 1, size(mappixels)
        if (hits(i) .gt. 0) then
            mappixels(i) = mappixels(i) / hits(i)
        end if
    end do

  end subroutine binned_map
  
